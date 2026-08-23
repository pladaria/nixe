//! Retained canonical RAM backing.

use std::collections::{BTreeMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{
    BackingIdentityExhausted, BackingStoreId, CanonicalBackingRange, CanonicalBackingSegment,
    CanonicalPageId, ContentGeneration, ContentMutationEpoch, CpuVisibilityRequest, CpuWriteEpoch,
    DeviceAccessDeclaration, DeviceVisibilityRequest, GenerationExhausted, GuestPhysicalPageId,
    MappingGeneration, MemoryPermissions, NonCpuDeviceId, VisibilityCoordinator, VisibilityError,
    VisibilityState,
};

struct CanonicalBackingStoreInner {
    identity: BackingStoreId,
    content_epoch: AtomicU64,
    cpu_write_epoch: AtomicU64,
    mutation: Mutex<CanonicalStoreMutationState>,
}

#[derive(Debug, Default)]
struct CanonicalStoreMutationState {
    cpu_writes: StoreCpuWriteJournal,
}

pub(crate) const STORE_CPU_WRITE_JOURNAL_CAPACITY: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StoreCpuWriteRecord {
    epoch: CpuWriteEpoch,
    page: CanonicalPageId,
    start: u64,
    end: u64,
}

#[derive(Debug, Default)]
struct StoreCpuWriteJournal {
    records: VecDeque<StoreCpuWriteRecord>,
    lost_through: Option<CpuWriteEpoch>,
}

impl StoreCpuWriteJournal {
    fn prepare(&mut self) -> Result<(), CanonicalContentMutationError> {
        if self.records.capacity() < STORE_CPU_WRITE_JOURNAL_CAPACITY {
            self.records
                .try_reserve_exact(STORE_CPU_WRITE_JOURNAL_CAPACITY - self.records.len())
                .map_err(|_| CanonicalContentMutationError::ResourceExhausted)?;
        }
        Ok(())
    }

    fn record(&mut self, epoch: CpuWriteEpoch, page: CanonicalPageId, start: u64, end: u64) {
        if start == end {
            return;
        }
        if self.records.len() == STORE_CPU_WRITE_JOURNAL_CAPACITY {
            let removed = self
                .records
                .pop_front()
                .expect("a full store CPU-write journal has a front record");
            self.lost_through = Some(
                self.lost_through
                    .map_or(removed.epoch, |current| current.max(removed.epoch)),
            );
        }
        self.records.push_back(StoreCpuWriteRecord {
            epoch,
            page,
            start,
            end,
        });
    }

    fn overlap_since(
        &self,
        epoch: CpuWriteEpoch,
        intervals: &[(CanonicalPageId, u64, u64)],
    ) -> CanonicalCpuWriteOverlap {
        if self
            .lost_through
            .is_some_and(|lost_through| epoch < lost_through)
        {
            return CanonicalCpuWriteOverlap::Unknown;
        }
        if self
            .records
            .iter()
            .rev()
            .take_while(|record| record.epoch > epoch)
            .any(|record| store_record_overlaps_intervals(*record, intervals))
        {
            CanonicalCpuWriteOverlap::Yes
        } else {
            CanonicalCpuWriteOverlap::No
        }
    }
}

fn store_record_overlaps_intervals(
    record: StoreCpuWriteRecord,
    intervals: &[(CanonicalPageId, u64, u64)],
) -> bool {
    let mut index = intervals.partition_point(|(page, _, _)| *page < record.page);
    while let Some((page, start, end)) = intervals.get(index) {
        if *page != record.page {
            break;
        }
        if record.start < *end && *start < record.end {
            return true;
        }
        index += 1;
    }
    false
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CanonicalContentMutationError {
    GenerationExhausted(GenerationExhausted),
    ResourceExhausted,
}

/// Shared authority for every canonical page in one backing store.
///
/// Per-page content generations remain authoritative for aliases and retained
/// ranges. This store-wide epoch is only a monotonic O(1) change detector for
/// CPU-engine memory synchronization.
#[derive(Clone)]
pub struct CanonicalBackingStore {
    inner: Arc<CanonicalBackingStoreInner>,
}

impl CanonicalBackingStore {
    /// Allocates a new globally unambiguous backing store.
    pub fn allocate() -> Result<Self, BackingIdentityExhausted> {
        Ok(Self {
            inner: Arc::new(CanonicalBackingStoreInner {
                identity: BackingStoreId::allocate()?,
                content_epoch: AtomicU64::new(ContentMutationEpoch::INITIAL.get()),
                cpu_write_epoch: AtomicU64::new(CpuWriteEpoch::INITIAL.get()),
                mutation: Mutex::new(CanonicalStoreMutationState::default()),
            }),
        })
    }

    /// Returns the stable pointer-free store identity.
    #[must_use]
    pub fn identity(&self) -> BackingStoreId {
        self.inner.identity
    }

    /// Returns the latest completely published content mutation in O(1).
    #[must_use]
    pub fn content_epoch(&self) -> ContentMutationEpoch {
        ContentMutationEpoch::new(self.inner.content_epoch.load(Ordering::Acquire))
    }

    /// Returns the latest completely published CPU-originated write in O(1).
    #[must_use]
    pub fn cpu_write_epoch(&self) -> CpuWriteEpoch {
        CpuWriteEpoch::new(self.inner.cpu_write_epoch.load(Ordering::Acquire))
    }

    fn begin_content_mutation(
        &self,
        cpu_write: bool,
    ) -> Result<CanonicalContentMutation<'_>, CanonicalContentMutationError> {
        let mut guard = self
            .inner
            .mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let next = self
            .content_epoch()
            .next()
            .map_err(CanonicalContentMutationError::GenerationExhausted)?;
        let next_cpu_write = cpu_write
            .then(|| self.cpu_write_epoch().next())
            .transpose()
            .map_err(CanonicalContentMutationError::GenerationExhausted)?;
        if cpu_write {
            guard.cpu_writes.prepare()?;
        }
        Ok(CanonicalContentMutation {
            store: self,
            guard,
            next,
            next_cpu_write,
        })
    }

    fn begin_cpu_write(
        &self,
    ) -> Result<CanonicalContentMutation<'_>, CanonicalContentMutationError> {
        self.begin_content_mutation(true)
    }

    fn begin_device_write(&self) -> Result<CanonicalContentMutation<'_>, GenerationExhausted> {
        self.begin_content_mutation(false)
            .map_err(|error| match error {
                CanonicalContentMutationError::GenerationExhausted(error) => error,
                CanonicalContentMutationError::ResourceExhausted => {
                    unreachable!("device mutations do not reserve CPU-write journal storage")
                }
            })
    }

    pub(crate) fn cpu_write_overlap_since(
        &self,
        epoch: CpuWriteEpoch,
        intervals: &[(CanonicalPageId, u64, u64)],
    ) -> (CpuWriteEpoch, CanonicalCpuWriteOverlap) {
        let mutation = self
            .inner
            .mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let current = self.cpu_write_epoch();
        let overlap = mutation.cpu_writes.overlap_since(epoch, intervals);
        (current, overlap)
    }
}

impl std::fmt::Debug for CanonicalBackingStore {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CanonicalBackingStore")
            .field("identity", &self.identity())
            .field("content_epoch", &self.content_epoch())
            .field("cpu_write_epoch", &self.cpu_write_epoch())
            .finish()
    }
}

struct CanonicalContentMutation<'a> {
    store: &'a CanonicalBackingStore,
    guard: std::sync::MutexGuard<'a, CanonicalStoreMutationState>,
    next: ContentMutationEpoch,
    next_cpu_write: Option<CpuWriteEpoch>,
}

impl CanonicalContentMutation<'_> {
    fn record_cpu_write(&mut self, page: CanonicalPageId, start: u64, end: u64) {
        if let Some(epoch) = self.next_cpu_write {
            self.guard.cpu_writes.record(epoch, page, start, end);
        }
    }

    fn commit(self) {
        self.store
            .inner
            .content_epoch
            .store(self.next.get(), Ordering::Release);
        if let Some(next) = self.next_cpu_write {
            self.store
                .inner
                .cpu_write_epoch
                .store(next.get(), Ordering::Release);
        }
        drop(self.guard);
    }
}

enum PageVisibility {
    Clean,
    CpuNewer,
    GpuNewer {
        device: NonCpuDeviceId,
        visible_at: crate::DeviceVisibilityPoint,
        coordinator: Arc<dyn VisibilityCoordinator>,
    },
    Conflicting,
    Invalid,
}

struct CanonicalPageState {
    bytes: Option<Box<[u8]>>,
    generation: ContentGeneration,
    visibility: PageVisibility,
    visibility_epoch: u64,
    cpu_writes: CpuWriteJournal,
}

const CPU_WRITE_JOURNAL_CAPACITY: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CpuWriteRecord {
    generation: ContentGeneration,
    start: u64,
    end: u64,
}

#[derive(Debug, Default)]
struct CpuWriteJournal {
    records: VecDeque<CpuWriteRecord>,
    lost_through: Option<ContentGeneration>,
}

impl CpuWriteJournal {
    fn prepare(&mut self) -> Result<(), CanonicalPageError> {
        if self.records.capacity() < CPU_WRITE_JOURNAL_CAPACITY {
            self.records
                .try_reserve_exact(CPU_WRITE_JOURNAL_CAPACITY - self.records.len())
                .map_err(|_| CanonicalPageError::ResourceExhausted)?;
        }
        Ok(())
    }

    fn record(&mut self, generation: ContentGeneration, start: u64, end: u64) {
        if start == end {
            return;
        }
        if self.records.len() == CPU_WRITE_JOURNAL_CAPACITY {
            let removed = self
                .records
                .pop_front()
                .expect("a full CPU-write journal has a front record");
            self.lost_through = Some(self.lost_through.map_or(removed.generation, |current| {
                current.max(removed.generation)
            }));
        }
        self.records.push_back(CpuWriteRecord {
            generation,
            start,
            end,
        });
    }

    fn overlap_since(
        &self,
        generation: ContentGeneration,
        start: u64,
        end: u64,
    ) -> CanonicalCpuWriteOverlap {
        if self
            .lost_through
            .is_some_and(|lost_through| generation < lost_through)
        {
            return CanonicalCpuWriteOverlap::Unknown;
        }
        if self.records.iter().any(|record| {
            record.generation > generation && record.start < end && start < record.end
        }) {
            CanonicalCpuWriteOverlap::Yes
        } else {
            CanonicalCpuWriteOverlap::No
        }
    }

    fn append_ranges_since(
        &self,
        generation: ContentGeneration,
        start: u64,
        end: u64,
        output: &mut Vec<CanonicalCpuWriteRange>,
    ) {
        if self
            .lost_through
            .is_some_and(|lost_through| generation < lost_through)
        {
            output.push(CanonicalCpuWriteRange::new(start, end - start));
            return;
        }
        for record in self
            .records
            .iter()
            .filter(|record| record.generation > generation)
        {
            let dirty_start = record.start.max(start);
            let dirty_end = record.end.min(end);
            if dirty_start < dirty_end {
                output.push(CanonicalCpuWriteRange::new(
                    dirty_start,
                    dirty_end - dirty_start,
                ));
            }
        }
    }
}

/// Whether canonical CPU-side writes touched a byte range after a snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalCpuWriteOverlap {
    No,
    Yes,
    /// The bounded provenance journal no longer covers the requested snapshot.
    Unknown,
}

/// One non-empty CPU-written byte interval within a canonical page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalCpuWriteRange {
    offset: u64,
    size: u64,
}

impl CanonicalCpuWriteRange {
    pub(crate) const fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// Returns the first dirty byte within the canonical page.
    #[must_use]
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Returns the number of dirty bytes.
    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }
}

struct CanonicalPageInner {
    store: CanonicalBackingStore,
    identity: CanonicalPageId,
    size: usize,
    state: Mutex<CanonicalPageState>,
}

/// A retained canonical RAM page.
///
/// Clones retain the bytes independently of CPU mappings and process-owned
/// page tables. Storage and host synchronization details remain private.
#[derive(Clone)]
pub struct CanonicalBackingPage {
    inner: Arc<CanonicalPageInner>,
}

impl std::fmt::Debug for CanonicalBackingPage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CanonicalBackingPage")
            .field("identity", &self.identity())
            .field("size", &self.size())
            .field("content_generation", &self.content_generation())
            .field("visibility", &self.visibility_state())
            .finish()
    }
}

impl PartialEq for CanonicalBackingPage {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for CanonicalBackingPage {}

impl CanonicalBackingPage {
    /// Creates a lazily materialized, zero-filled canonical page.
    pub fn zeroed(
        store: &CanonicalBackingStore,
        page: GuestPhysicalPageId,
        size: usize,
        generation: ContentGeneration,
    ) -> Result<Self, CanonicalPageError> {
        if size == 0 {
            return Err(CanonicalPageError::InvalidSize);
        }
        Ok(Self {
            inner: Arc::new(CanonicalPageInner {
                store: store.clone(),
                identity: CanonicalPageId::new(store.identity(), page),
                size,
                state: Mutex::new(CanonicalPageState {
                    bytes: None,
                    generation,
                    visibility: PageVisibility::Clean,
                    visibility_epoch: 0,
                    cpu_writes: CpuWriteJournal::default(),
                }),
            }),
        })
    }

    /// Creates a canonical page initialized from exactly one page of bytes.
    pub fn initialized(
        store: &CanonicalBackingStore,
        page: GuestPhysicalPageId,
        bytes: &[u8],
        generation: ContentGeneration,
    ) -> Result<Self, CanonicalPageError> {
        if bytes.is_empty() {
            return Err(CanonicalPageError::InvalidSize);
        }
        let mut contents = Vec::new();
        contents
            .try_reserve_exact(bytes.len())
            .map_err(|_| CanonicalPageError::ResourceExhausted)?;
        contents.extend_from_slice(bytes);
        Ok(Self {
            inner: Arc::new(CanonicalPageInner {
                store: store.clone(),
                identity: CanonicalPageId::new(store.identity(), page),
                size: bytes.len(),
                state: Mutex::new(CanonicalPageState {
                    bytes: Some(contents.into_boxed_slice()),
                    generation,
                    visibility: PageVisibility::Clean,
                    visibility_epoch: 0,
                    cpu_writes: CpuWriteJournal::default(),
                }),
            }),
        })
    }

    /// Returns the stable cross-device page identity.
    #[must_use]
    pub fn identity(&self) -> CanonicalPageId {
        self.inner.identity
    }

    pub(crate) fn store(&self) -> &CanonicalBackingStore {
        &self.inner.store
    }

    pub(crate) fn cpu_write_epoch(&self) -> CpuWriteEpoch {
        self.store().cpu_write_epoch()
    }

    /// Returns the byte size of this backing page.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.size
    }

    /// Returns the current byte-content generation.
    #[must_use]
    pub fn content_generation(&self) -> ContentGeneration {
        self.lock_state().generation
    }

    /// Returns the conservative authority state shared by every page alias.
    #[must_use]
    pub fn visibility_state(&self) -> VisibilityState {
        Self::visibility_snapshot(&self.lock_state().visibility)
    }

    /// Copies a checked byte range out of canonical storage.
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<(), CanonicalPageError> {
        self.read_with_generation(offset, output).map(|_| ())
    }

    /// Copies bytes and observes their content generation under the same page
    /// lock. Exclusive-load adapters use this to avoid pairing stale bytes with
    /// a newer reservation generation.
    pub fn read_with_generation(
        &self,
        offset: usize,
        output: &mut [u8],
    ) -> Result<ContentGeneration, CanonicalPageError> {
        let end = self.checked_end(offset, output.len())?;
        loop {
            let state = self.lock_state();
            match state.visibility {
                PageVisibility::Clean | PageVisibility::CpuNewer => {
                    if let Some(bytes) = &state.bytes {
                        output.copy_from_slice(&bytes[offset..end]);
                    } else {
                        output.fill(0);
                    }
                    return Ok(state.generation);
                }
                PageVisibility::GpuNewer { .. } => {
                    // Device reconciliation cannot run while the observation
                    // lock is held. CPU-visible pages complete above with one
                    // lock acquisition; only this uncommon path retries.
                    drop(state);
                    self.ensure_cpu_visible()
                        .map_err(CanonicalPageError::Visibility)?;
                }
                PageVisibility::Conflicting => {
                    return Err(CanonicalPageError::Visibility(
                        VisibilityError::ConflictingAccess,
                    ));
                }
                PageVisibility::Invalid => {
                    return Err(CanonicalPageError::Visibility(
                        VisibilityError::InvalidState,
                    ));
                }
            }
        }
    }

    fn snapshot_cpu_write(
        &self,
    ) -> Result<(Box<[u8]>, ContentGeneration, u64), CanonicalPageError> {
        loop {
            self.ensure_cpu_visible()
                .map_err(CanonicalPageError::Visibility)?;
            let mut state = self.lock_state();
            match state.visibility {
                PageVisibility::Clean | PageVisibility::CpuNewer => {
                    state.cpu_writes.prepare()?;
                    let mut bytes = Vec::new();
                    bytes
                        .try_reserve_exact(self.size())
                        .map_err(|_| CanonicalPageError::ResourceExhausted)?;
                    if let Some(contents) = &state.bytes {
                        bytes.extend_from_slice(contents);
                    } else {
                        bytes.resize(self.size(), 0);
                    }
                    return Ok((
                        bytes.into_boxed_slice(),
                        state.generation,
                        state.visibility_epoch,
                    ));
                }
                PageVisibility::GpuNewer { .. } => {}
                PageVisibility::Conflicting => {
                    return Err(CanonicalPageError::Visibility(
                        VisibilityError::ConflictingAccess,
                    ));
                }
                PageVisibility::Invalid => {
                    return Err(CanonicalPageError::Visibility(
                        VisibilityError::InvalidState,
                    ));
                }
            }
        }
    }

    /// Materializes zero storage before a multi-page write is published.
    ///
    /// Allocation may fail, but a successful call does not change guest bytes
    /// or their content generation.
    pub fn prepare_write(&self) -> Result<(), CanonicalPageError> {
        self.prepare_cpu_access()?;
        let mut state = self.lock_state();
        Self::require_cpu_authority(&mut state).map_err(CanonicalPageError::Visibility)?;
        if state.bytes.is_none() {
            let mut contents = Vec::new();
            contents
                .try_reserve_exact(self.inner.size)
                .map_err(|_| CanonicalPageError::ResourceExhausted)?;
            contents.resize(self.inner.size, 0);
            state.bytes = Some(contents.into_boxed_slice());
        }
        state.cpu_writes.prepare()?;
        Ok(())
    }

    /// Establishes canonical CPU visibility without reading or modifying bytes.
    ///
    /// CPU adapters use this before operations, such as exclusive stores,
    /// whose success depends on observing the newest content generation.
    pub fn prepare_cpu_access(&self) -> Result<(), CanonicalPageError> {
        self.ensure_cpu_visible()
            .map_err(CanonicalPageError::Visibility)
    }

    /// Atomically writes bytes and publishes a preflighted next generation.
    ///
    /// The expected generation prevents a retained observer from committing a
    /// write using stale preflight state.
    pub fn write_preflighted(
        &self,
        offset: usize,
        bytes: &[u8],
        expected: ContentGeneration,
        next: ContentGeneration,
    ) -> Result<(), CanonicalPageError> {
        let end = self.checked_end(offset, bytes.len())?;
        if expected.next() != Ok(next) {
            return Err(CanonicalPageError::InvalidGenerationTransition);
        }
        let mut mutation = self
            .store()
            .begin_cpu_write()
            .map_err(|error| match error {
                CanonicalContentMutationError::GenerationExhausted(error) => {
                    CanonicalPageError::MutationEpochExhausted(error)
                }
                CanonicalContentMutationError::ResourceExhausted => {
                    CanonicalPageError::ResourceExhausted
                }
            })?;
        let mut state = self.lock_state();
        state.cpu_writes.prepare()?;
        if state.generation != expected {
            return Err(CanonicalPageError::StaleGeneration {
                expected,
                observed: state.generation,
            });
        }
        Self::require_cpu_authority(&mut state).map_err(CanonicalPageError::Visibility)?;
        Self::publish_visibility(&mut state, PageVisibility::CpuNewer)
            .map_err(CanonicalPageError::Visibility)?;
        if !bytes.is_empty() {
            if state.bytes.is_none() {
                let mut contents = Vec::new();
                contents
                    .try_reserve_exact(self.inner.size)
                    .map_err(|_| CanonicalPageError::ResourceExhausted)?;
                contents.resize(self.inner.size, 0);
                state.bytes = Some(contents.into_boxed_slice());
            }
            state
                .bytes
                .as_mut()
                .expect("materialized canonical page has bytes")[offset..end]
                .copy_from_slice(bytes);
        }
        state.generation = next;
        state.cpu_writes.record(next, offset as u64, end as u64);
        mutation.record_cpu_write(self.identity(), offset as u64, end as u64);
        mutation.commit();
        Ok(())
    }

    /// Writes another fragment of an already-published logical mutation.
    ///
    /// This is used only to commit a page-spanning operation which preflighted
    /// one generation per distinct page. The current generation must equal
    /// the operation's published generation.
    pub fn write_fragment_preflighted(
        &self,
        offset: usize,
        bytes: &[u8],
        generation: ContentGeneration,
    ) -> Result<(), CanonicalPageError> {
        let end = self.checked_end(offset, bytes.len())?;
        let mut mutation = self
            .store()
            .begin_cpu_write()
            .map_err(|error| match error {
                CanonicalContentMutationError::GenerationExhausted(error) => {
                    CanonicalPageError::MutationEpochExhausted(error)
                }
                CanonicalContentMutationError::ResourceExhausted => {
                    CanonicalPageError::ResourceExhausted
                }
            })?;
        let mut state = self.lock_state();
        state.cpu_writes.prepare()?;
        if state.generation != generation {
            return Err(CanonicalPageError::StaleGeneration {
                expected: generation,
                observed: state.generation,
            });
        }
        Self::require_cpu_authority(&mut state).map_err(CanonicalPageError::Visibility)?;
        Self::publish_visibility(&mut state, PageVisibility::CpuNewer)
            .map_err(CanonicalPageError::Visibility)?;
        if !bytes.is_empty() {
            if state.bytes.is_none() {
                let mut contents = Vec::new();
                contents
                    .try_reserve_exact(self.inner.size)
                    .map_err(|_| CanonicalPageError::ResourceExhausted)?;
                contents.resize(self.inner.size, 0);
                state.bytes = Some(contents.into_boxed_slice());
            }
            state
                .bytes
                .as_mut()
                .expect("materialized canonical page has bytes")[offset..end]
                .copy_from_slice(bytes);
        }
        state
            .cpu_writes
            .record(generation, offset as u64, end as u64);
        mutation.record_cpu_write(self.identity(), offset as u64, end as u64);
        mutation.commit();
        Ok(())
    }

    /// Reports exact CPU-side byte writes newer than `generation`.
    pub fn cpu_write_overlap_since(
        &self,
        generation: ContentGeneration,
        offset: u64,
        size: u64,
    ) -> Result<CanonicalCpuWriteOverlap, CanonicalPageError> {
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalPageError::InvalidRange)?;
        if end > self.size() as u64 {
            return Err(CanonicalPageError::InvalidRange);
        }
        Ok(self
            .lock_state()
            .cpu_writes
            .overlap_since(generation, offset, end))
    }

    /// Appends CPU-written byte intervals newer than `generation`.
    ///
    /// Intervals use page-local offsets. When bounded provenance has expired,
    /// the complete requested interval is appended so callers cannot miss a
    /// write.
    pub fn append_cpu_write_ranges_since(
        &self,
        generation: ContentGeneration,
        offset: u64,
        size: u64,
        output: &mut Vec<CanonicalCpuWriteRange>,
    ) -> Result<(), CanonicalPageError> {
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalPageError::InvalidRange)?;
        if end > self.size() as u64 {
            return Err(CanonicalPageError::InvalidRange);
        }
        self.lock_state()
            .cpu_writes
            .append_ranges_since(generation, offset, end, output);
        Ok(())
    }

    pub(crate) fn prepare_device_access(
        &self,
        declaration: DeviceAccessDeclaration,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<(), VisibilityError> {
        let target = declaration.device_visible_at();
        let (bytes, generation, epoch) = {
            let mut state = self.lock_state();
            match &state.visibility {
                PageVisibility::GpuNewer {
                    device, visible_at, ..
                } if *device == declaration.device() && *visible_at <= target => return Ok(()),
                PageVisibility::GpuNewer { .. } | PageVisibility::Conflicting => {
                    Self::publish_visibility(&mut state, PageVisibility::Conflicting)?;
                    return Err(VisibilityError::ConflictingAccess);
                }
                PageVisibility::Invalid => return Err(VisibilityError::InvalidState),
                PageVisibility::Clean | PageVisibility::CpuNewer => {}
            }
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(self.inner.size)
                .map_err(|_| VisibilityError::ResourceExhausted)?;
            if let Some(contents) = &state.bytes {
                bytes.extend_from_slice(contents);
            } else {
                bytes.resize(self.inner.size, 0);
            }
            (bytes, state.generation, state.visibility_epoch)
        };
        let request = DeviceVisibilityRequest {
            page: self.identity(),
            size: self.size(),
            device: declaration.device(),
            visible_at: target,
        };
        if let Err(error) = coordinator.make_device_visible(request, &bytes) {
            let mut state = self.lock_state();
            Self::publish_visibility(&mut state, PageVisibility::Invalid)?;
            return Err(VisibilityError::Coordinator(error));
        }
        let mut state = self.lock_state();
        if state.visibility_epoch != epoch || state.generation != generation {
            Self::publish_visibility(&mut state, PageVisibility::Conflicting)?;
            return Err(VisibilityError::ConcurrentTransition);
        }
        Self::publish_visibility(&mut state, PageVisibility::Clean)
    }

    pub(crate) fn prepare_resident_device_read(
        &self,
        declaration: DeviceAccessDeclaration,
    ) -> Result<(), VisibilityError> {
        let mut state = self.lock_state();
        match state.visibility {
            PageVisibility::Clean => Ok(()),
            PageVisibility::CpuNewer => Self::publish_visibility(&mut state, PageVisibility::Clean),
            PageVisibility::GpuNewer {
                device, visible_at, ..
            } if device == declaration.device()
                && visible_at <= declaration.device_visible_at() =>
            {
                Ok(())
            }
            PageVisibility::GpuNewer { .. } | PageVisibility::Conflicting => {
                Self::publish_visibility(&mut state, PageVisibility::Conflicting)?;
                Err(VisibilityError::ConflictingAccess)
            }
            PageVisibility::Invalid => Err(VisibilityError::InvalidState),
        }
    }

    pub(crate) fn publish_device_write(
        &self,
        declaration: DeviceAccessDeclaration,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<(), VisibilityError> {
        let Some(visible_at) = declaration.cpu_visible_at() else {
            return Err(VisibilityError::DeclarationDoesNotWrite);
        };
        let mutation = self
            .store()
            .begin_device_write()
            .map_err(VisibilityError::GenerationExhausted)?;
        let mut state = self.lock_state();
        match &state.visibility {
            PageVisibility::Clean => {}
            PageVisibility::GpuNewer {
                device,
                visible_at: previous,
                ..
            } if *device == declaration.device() && *previous <= visible_at => {}
            PageVisibility::CpuNewer
            | PageVisibility::GpuNewer { .. }
            | PageVisibility::Conflicting => {
                Self::publish_visibility(&mut state, PageVisibility::Conflicting)?;
                return Err(VisibilityError::ConflictingAccess);
            }
            PageVisibility::Invalid => return Err(VisibilityError::InvalidState),
        }
        Self::publish_visibility(
            &mut state,
            PageVisibility::GpuNewer {
                device: declaration.device(),
                visible_at,
                coordinator,
            },
        )?;
        mutation.commit();
        Ok(())
    }

    pub(crate) fn invalidate_visibility(&self) -> Result<(), VisibilityError> {
        Self::publish_visibility(&mut self.lock_state(), PageVisibility::Invalid)
    }

    fn ensure_cpu_visible(&self) -> Result<(), VisibilityError> {
        let (device, visible_at, coordinator, epoch, next_generation) = {
            let mut state = self.lock_state();
            match &state.visibility {
                PageVisibility::Clean | PageVisibility::CpuNewer => return Ok(()),
                PageVisibility::Conflicting => return Err(VisibilityError::ConflictingAccess),
                PageVisibility::Invalid => return Err(VisibilityError::InvalidState),
                PageVisibility::GpuNewer {
                    device,
                    visible_at,
                    coordinator,
                } => {
                    let next = match state.generation.next() {
                        Ok(next) => next,
                        Err(error) => {
                            Self::publish_visibility(&mut state, PageVisibility::Invalid)?;
                            return Err(VisibilityError::GenerationExhausted(error));
                        }
                    };
                    (
                        *device,
                        *visible_at,
                        Arc::clone(coordinator),
                        state.visibility_epoch,
                        next,
                    )
                }
            }
        };
        let request = CpuVisibilityRequest {
            page: self.identity(),
            size: self.size(),
            device,
            visible_at,
        };
        let bytes = match coordinator.make_cpu_visible(request) {
            Ok(bytes) => bytes,
            Err(error) => {
                let mut state = self.lock_state();
                Self::publish_visibility(&mut state, PageVisibility::Invalid)?;
                return Err(VisibilityError::Coordinator(error));
            }
        };
        if bytes.len() != self.size() {
            let observed = bytes.len();
            let mut state = self.lock_state();
            Self::publish_visibility(&mut state, PageVisibility::Invalid)?;
            return Err(VisibilityError::IncorrectWritebackSize {
                expected: self.size(),
                observed,
            });
        }
        let mut state = self.lock_state();
        if state.visibility_epoch != epoch {
            Self::publish_visibility(&mut state, PageVisibility::Conflicting)?;
            return Err(VisibilityError::ConcurrentTransition);
        }
        Self::publish_visibility(&mut state, PageVisibility::Clean)?;
        state.bytes = Some(bytes);
        state.generation = next_generation;
        Ok(())
    }

    fn visibility_snapshot(visibility: &PageVisibility) -> VisibilityState {
        match visibility {
            PageVisibility::Clean => VisibilityState::Clean,
            PageVisibility::CpuNewer => VisibilityState::CpuNewer,
            PageVisibility::GpuNewer {
                device, visible_at, ..
            } => VisibilityState::GpuNewer {
                device: *device,
                visible_at: *visible_at,
            },
            PageVisibility::Conflicting => VisibilityState::Conflicting,
            PageVisibility::Invalid => VisibilityState::Invalid,
        }
    }

    fn require_cpu_authority(state: &mut CanonicalPageState) -> Result<(), VisibilityError> {
        match state.visibility {
            PageVisibility::Clean | PageVisibility::CpuNewer => Ok(()),
            PageVisibility::GpuNewer { .. } | PageVisibility::Conflicting => {
                Self::publish_visibility(state, PageVisibility::Conflicting)?;
                Err(VisibilityError::ConflictingAccess)
            }
            PageVisibility::Invalid => Err(VisibilityError::InvalidState),
        }
    }

    fn publish_visibility(
        state: &mut CanonicalPageState,
        visibility: PageVisibility,
    ) -> Result<(), VisibilityError> {
        let Some(next_epoch) = state.visibility_epoch.checked_add(1) else {
            state.visibility = PageVisibility::Invalid;
            return Err(VisibilityError::VisibilityEpochExhausted);
        };
        state.visibility = visibility;
        state.visibility_epoch = next_epoch;
        Ok(())
    }

    fn checked_end(&self, offset: usize, size: usize) -> Result<usize, CanonicalPageError> {
        offset
            .checked_add(size)
            .filter(|end| *end <= self.inner.size)
            .ok_or(CanonicalPageError::InvalidRange)
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, CanonicalPageState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

struct PendingCanonicalPageWrite {
    backing: CanonicalBackingPage,
    expected_generation: ContentGeneration,
    expected_visibility_epoch: u64,
    bytes: Box<[u8]>,
    dirty_ranges: Vec<(u64, u64)>,
}

/// Atomic CPU-side mutation assembled over retained canonical ranges.
///
/// This is intended for deterministic software device interpreters. Staging
/// may establish CPU visibility but does not alter bytes or generations. A
/// successful commit locks every affected page in identity order, validates
/// all snapshots, and publishes the complete batch at once.
#[derive(Default)]
pub struct CanonicalWriteBatch {
    pages: BTreeMap<CanonicalPageId, PendingCanonicalPageWrite>,
}

impl CanonicalWriteBatch {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pages: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Reads canonical bytes with this transaction's earlier writes overlaid.
    ///
    /// Ordered device operations can therefore consume preceding writes while
    /// the complete batch remains unpublished and atomically discardable.
    pub fn read_staged(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        output: &mut [u8],
    ) -> Result<(), CanonicalWriteBatchError> {
        let size =
            u64::try_from(output.len()).map_err(|_| CanonicalWriteBatchError::RangeOverflow)?;
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
        if end > range.size() {
            return Err(CanonicalWriteBatchError::OutOfBounds {
                offset,
                size,
                range_size: range.size(),
            });
        }
        if output.is_empty() {
            return Ok(());
        }

        let mut logical_start = 0_u64;
        let mut copied = 0_usize;
        for segment in range.segments() {
            let logical_end = logical_start
                .checked_add(segment.size())
                .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
            let read_start = offset.max(logical_start);
            let read_end = end.min(logical_end);
            if read_start < read_end {
                if !segment.permissions().contains(MemoryPermissions::READ) {
                    return Err(CanonicalWriteBatchError::PermissionDenied {
                        page: segment.page(),
                        available: segment.permissions(),
                    });
                }
                let within_segment = read_start - logical_start;
                let page_offset = segment
                    .offset()
                    .checked_add(within_segment)
                    .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
                let page_offset = usize::try_from(page_offset)
                    .map_err(|_| CanonicalWriteBatchError::RangeOverflow)?;
                let count = usize::try_from(read_end - read_start)
                    .map_err(|_| CanonicalWriteBatchError::RangeOverflow)?;
                let copied_end = copied
                    .checked_add(count)
                    .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
                if let Some(pending) = self.pages.get(&segment.page()) {
                    output[copied..copied_end]
                        .copy_from_slice(&pending.bytes[page_offset..page_offset + count]);
                } else {
                    segment
                        .backing()
                        .read(page_offset, &mut output[copied..copied_end])
                        .map_err(CanonicalWriteBatchError::Page)?;
                }
                copied = copied_end;
            }
            logical_start = logical_end;
            if logical_start >= end {
                break;
            }
        }
        if copied != output.len() {
            return Err(CanonicalWriteBatchError::IncompleteRange);
        }
        Ok(())
    }

    /// Stages one checked logical write. Discard the batch if this fails.
    pub fn stage(
        &mut self,
        range: &CanonicalBackingRange,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), CanonicalWriteBatchError> {
        let size =
            u64::try_from(bytes.len()).map_err(|_| CanonicalWriteBatchError::RangeOverflow)?;
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
        if end > range.size() {
            return Err(CanonicalWriteBatchError::OutOfBounds {
                offset,
                size,
                range_size: range.size(),
            });
        }
        if bytes.is_empty() {
            return Ok(());
        }

        let mut logical_start = 0_u64;
        let mut copied = 0_usize;
        for segment in range.segments() {
            let logical_end = logical_start
                .checked_add(segment.size())
                .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
            let write_start = offset.max(logical_start);
            let write_end = end.min(logical_end);
            if write_start < write_end {
                if !segment.permissions().contains(MemoryPermissions::WRITE) {
                    return Err(CanonicalWriteBatchError::PermissionDenied {
                        page: segment.page(),
                        available: segment.permissions(),
                    });
                }
                let pending = match self.pages.entry(segment.page()) {
                    std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        let (page_bytes, generation, visibility_epoch) = segment
                            .backing()
                            .snapshot_cpu_write()
                            .map_err(CanonicalWriteBatchError::Page)?;
                        entry.insert(PendingCanonicalPageWrite {
                            backing: segment.backing().clone(),
                            expected_generation: generation,
                            expected_visibility_epoch: visibility_epoch,
                            bytes: page_bytes,
                            dirty_ranges: Vec::new(),
                        })
                    }
                };
                let within_segment = write_start - logical_start;
                let page_offset = segment
                    .offset()
                    .checked_add(within_segment)
                    .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
                let page_offset = usize::try_from(page_offset)
                    .map_err(|_| CanonicalWriteBatchError::RangeOverflow)?;
                let count = usize::try_from(write_end - write_start)
                    .map_err(|_| CanonicalWriteBatchError::RangeOverflow)?;
                let copied_end = copied
                    .checked_add(count)
                    .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
                pending.bytes[page_offset..page_offset + count]
                    .copy_from_slice(&bytes[copied..copied_end]);
                pending
                    .dirty_ranges
                    .push((page_offset as u64, (page_offset + count) as u64));
                copied = copied_end;
            }
            logical_start = logical_end;
            if logical_start >= end {
                break;
            }
        }
        if copied != bytes.len() {
            return Err(CanonicalWriteBatchError::IncompleteRange);
        }
        Ok(())
    }

    /// Publishes every staged byte mutation and advances each page once.
    pub fn commit(self) -> Result<(), CanonicalWriteBatchError> {
        struct Write {
            expected_generation: ContentGeneration,
            expected_visibility_epoch: u64,
            next_generation: ContentGeneration,
            next_visibility_epoch: u64,
            bytes: Option<Box<[u8]>>,
            dirty_ranges: Vec<(u64, u64)>,
        }

        let mut backings = Vec::new();
        let mut writes = Vec::new();
        backings
            .try_reserve_exact(self.pages.len())
            .map_err(|_| CanonicalWriteBatchError::ResourceExhausted)?;
        writes
            .try_reserve_exact(self.pages.len())
            .map_err(|_| CanonicalWriteBatchError::ResourceExhausted)?;
        for pending in self.pages.into_values() {
            let next_generation = pending
                .expected_generation
                .next()
                .map_err(CanonicalWriteBatchError::GenerationExhausted)?;
            let next_visibility_epoch = pending
                .expected_visibility_epoch
                .checked_add(1)
                .ok_or(CanonicalWriteBatchError::VisibilityEpochExhausted)?;
            backings.push(pending.backing);
            writes.push(Write {
                expected_generation: pending.expected_generation,
                expected_visibility_epoch: pending.expected_visibility_epoch,
                next_generation,
                next_visibility_epoch,
                bytes: Some(pending.bytes),
                dirty_ranges: pending.dirty_ranges,
            });
        }

        let mut stores = backings
            .iter()
            .map(|backing| backing.store().clone())
            .collect::<Vec<_>>();
        stores.sort_by_key(CanonicalBackingStore::identity);
        stores.dedup_by_key(|store| store.identity());
        let mut mutations = stores
            .iter()
            .map(|store| {
                store.begin_cpu_write().map_err(|error| match error {
                    CanonicalContentMutationError::GenerationExhausted(error) => {
                        CanonicalWriteBatchError::GenerationExhausted(error)
                    }
                    CanonicalContentMutationError::ResourceExhausted => {
                        CanonicalWriteBatchError::ResourceExhausted
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut states = backings
            .iter()
            .map(CanonicalBackingPage::lock_state)
            .collect::<Vec<_>>();
        for (state, write) in states.iter().zip(&writes) {
            if state.generation != write.expected_generation
                || state.visibility_epoch != write.expected_visibility_epoch
                || !matches!(
                    state.visibility,
                    PageVisibility::Clean | PageVisibility::CpuNewer
                )
            {
                return Err(CanonicalWriteBatchError::ConcurrentMutation);
            }
        }
        for (state, write) in states.iter_mut().zip(&mut writes) {
            state.bytes = write.bytes.take();
            state.generation = write.next_generation;
            state.visibility = PageVisibility::CpuNewer;
            state.visibility_epoch = write.next_visibility_epoch;
            for &(start, end) in &write.dirty_ranges {
                state.cpu_writes.record(write.next_generation, start, end);
            }
        }
        for (backing, write) in backings.iter().zip(&writes) {
            let mutation = mutations
                .iter_mut()
                .find(|mutation| mutation.store.identity() == backing.identity().store())
                .expect("every written backing store has a prepared mutation");
            for &(start, end) in &write.dirty_ranges {
                mutation.record_cpu_write(backing.identity(), start, end);
            }
        }
        for mutation in mutations {
            mutation.commit();
        }
        Ok(())
    }
}

/// Failure while staging or atomically committing canonical writes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalWriteBatchError {
    RangeOverflow,
    OutOfBounds {
        offset: u64,
        size: u64,
        range_size: u64,
    },
    PermissionDenied {
        page: CanonicalPageId,
        available: MemoryPermissions,
    },
    IncompleteRange,
    Page(CanonicalPageError),
    GenerationExhausted(GenerationExhausted),
    VisibilityEpochExhausted,
    ConcurrentMutation,
    ResourceExhausted,
}

impl Display for CanonicalWriteBatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RangeOverflow => formatter.write_str("canonical write batch range overflows"),
            Self::OutOfBounds {
                offset,
                size,
                range_size,
            } => write!(
                formatter,
                "canonical write batch offset={offset:#x} size={size:#x} exceeds range-size={range_size:#x}"
            ),
            Self::PermissionDenied { page, available } => write!(
                formatter,
                "canonical write batch permission denied for {page}: required=0x{:x} available=0x{:x}",
                MemoryPermissions::WRITE.bits(),
                available.bits()
            ),
            Self::IncompleteRange => {
                formatter.write_str("canonical write batch range is incomplete")
            }
            Self::Page(error) => write!(
                formatter,
                "canonical write batch page access failed: {error}"
            ),
            Self::GenerationExhausted(error) => error.fmt(formatter),
            Self::VisibilityEpochExhausted => {
                formatter.write_str("canonical write batch visibility epoch is exhausted")
            }
            Self::ConcurrentMutation => {
                formatter.write_str("canonical bytes changed while a write batch was staged")
            }
            Self::ResourceExhausted => {
                formatter.write_str("canonical write batch exhausted host resources")
            }
        }
    }
}

impl std::error::Error for CanonicalWriteBatchError {}

/// Failure while creating or accessing canonical RAM backing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalPageError {
    /// A page cannot have zero bytes.
    InvalidSize,
    /// A requested subrange is outside the page.
    InvalidRange,
    /// Host allocation failed.
    ResourceExhausted,
    /// The supplied generation was no longer current.
    StaleGeneration {
        expected: ContentGeneration,
        observed: ContentGeneration,
    },
    /// The supplied next generation was not the exact successor.
    InvalidGenerationTransition,
    /// The store-wide content mutation epoch cannot advance without wrapping.
    MutationEpochExhausted(GenerationExhausted),
    /// Required CPU/device visibility work could not be completed.
    Visibility(VisibilityError),
}

impl Display for CanonicalPageError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSize => formatter.write_str("canonical page size is zero"),
            Self::InvalidRange => formatter.write_str("canonical page range is out of bounds"),
            Self::ResourceExhausted => {
                formatter.write_str("host resources for canonical backing are exhausted")
            }
            Self::StaleGeneration { expected, observed } => write!(
                formatter,
                "canonical page generation changed: expected {expected}, observed {observed}"
            ),
            Self::InvalidGenerationTransition => {
                formatter.write_str("canonical page generation transition is not consecutive")
            }
            Self::MutationEpochExhausted(error) => error.fmt(formatter),
            Self::Visibility(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CanonicalPageError {}

#[derive(Debug)]
struct CanonicalAllocationInner {
    store: CanonicalBackingStore,
    size: usize,
    page_size: usize,
    pages: Box<[CanonicalBackingPage]>,
    transaction: Mutex<()>,
}

/// Retained canonical allocation suitable for kernel objects and future
/// device mappings.
#[derive(Clone, Debug)]
pub struct CanonicalAllocation {
    inner: Arc<CanonicalAllocationInner>,
}

impl CanonicalAllocation {
    /// Creates a zero-filled allocation divided into fixed-size canonical pages.
    pub fn zeroed(size: usize, page_size: usize) -> Result<Self, CanonicalAllocationError> {
        if size == 0 || page_size == 0 || !page_size.is_power_of_two() {
            return Err(CanonicalAllocationError::InvalidSize);
        }
        let store = CanonicalBackingStore::allocate()
            .map_err(CanonicalAllocationError::IdentityExhausted)?;
        let page_count = size.div_ceil(page_size);
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(page_count)
            .map_err(|_| CanonicalAllocationError::ResourceExhausted)?;
        for index in 0..page_count {
            let local_id = u64::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .ok_or(CanonicalAllocationError::IdentityExhausted(
                    BackingIdentityExhausted,
                ))?;
            pages.push(
                CanonicalBackingPage::zeroed(
                    &store,
                    GuestPhysicalPageId::new(local_id),
                    page_size,
                    ContentGeneration::INITIAL,
                )
                .map_err(CanonicalAllocationError::Page)?,
            );
        }
        Ok(Self {
            inner: Arc::new(CanonicalAllocationInner {
                store,
                size,
                page_size,
                pages: pages.into_boxed_slice(),
                transaction: Mutex::new(()),
            }),
        })
    }

    /// Returns the stable ownership-domain identity.
    #[must_use]
    pub fn store(&self) -> BackingStoreId {
        self.inner.store.identity()
    }

    /// Returns the logical byte size, excluding padding in the final page.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.size
    }

    /// Returns whether two values retain the same allocation.
    #[must_use]
    pub fn same_backing(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Copies a checked logical range out of canonical backing.
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<(), CanonicalAllocationError> {
        let end = self.checked_end(offset, output.len())?;
        let _transaction = self.lock_transaction();
        let mut cursor = offset;
        let mut copied = 0;
        while cursor < end {
            let page_index = cursor / self.inner.page_size;
            let page_offset = cursor % self.inner.page_size;
            let count = (self.inner.page_size - page_offset).min(end - cursor);
            self.inner.pages[page_index]
                .read(page_offset, &mut output[copied..copied + count])
                .map_err(CanonicalAllocationError::Page)?;
            cursor += count;
            copied += count;
        }
        Ok(())
    }

    /// Atomically writes a checked logical range and advances each affected
    /// page generation once.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), CanonicalAllocationError> {
        let end = self.checked_end(offset, bytes.len())?;
        let _transaction = self.lock_transaction();
        if bytes.is_empty() {
            return Ok(());
        }
        let first_page = offset / self.inner.page_size;
        let last_page = (end - 1) / self.inner.page_size;
        let mut generations = Vec::new();
        generations
            .try_reserve_exact(last_page - first_page + 1)
            .map_err(|_| CanonicalAllocationError::ResourceExhausted)?;
        for page in &self.inner.pages[first_page..=last_page] {
            page.prepare_write()
                .map_err(CanonicalAllocationError::Page)?;
            let current = page.content_generation();
            let next = current
                .next()
                .map_err(CanonicalAllocationError::GenerationExhausted)?;
            generations.push((current, next));
        }
        let mut cursor = offset;
        let mut copied = 0;
        while cursor < end {
            let page_index = cursor / self.inner.page_size;
            let page_offset = cursor % self.inner.page_size;
            let count = (self.inner.page_size - page_offset).min(end - cursor);
            let (current, next) = generations[page_index - first_page];
            self.inner.pages[page_index]
                .write_preflighted(page_offset, &bytes[copied..copied + count], current, next)
                .map_err(CanonicalAllocationError::Page)?;
            cursor += count;
            copied += count;
        }
        Ok(())
    }

    /// Creates a retained pointer-free view over the complete allocation.
    pub fn backing_range(
        &self,
        permissions: MemoryPermissions,
    ) -> Result<CanonicalBackingRange, CanonicalAllocationError> {
        let _transaction = self.lock_transaction();
        let mut segments = Vec::new();
        segments
            .try_reserve_exact(self.inner.pages.len())
            .map_err(|_| CanonicalAllocationError::ResourceExhausted)?;
        let mut remaining = self.inner.size;
        for page in &self.inner.pages {
            let size = remaining.min(self.inner.page_size);
            segments.push(
                CanonicalBackingSegment::new(
                    page.clone(),
                    0,
                    size as u64,
                    permissions,
                    page.content_generation(),
                    MappingGeneration::INITIAL,
                )
                .map_err(CanonicalAllocationError::Range)?,
            );
            remaining -= size;
        }
        CanonicalBackingRange::new(segments).map_err(CanonicalAllocationError::Range)
    }

    fn checked_end(&self, offset: usize, size: usize) -> Result<usize, CanonicalAllocationError> {
        offset
            .checked_add(size)
            .filter(|end| *end <= self.inner.size)
            .ok_or(CanonicalAllocationError::InvalidRange)
    }

    fn lock_transaction(&self) -> std::sync::MutexGuard<'_, ()> {
        self.inner
            .transaction
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Failure while creating or accessing a retained canonical allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalAllocationError {
    InvalidSize,
    InvalidRange,
    ResourceExhausted,
    IdentityExhausted(BackingIdentityExhausted),
    GenerationExhausted(GenerationExhausted),
    Page(CanonicalPageError),
    Range(crate::CanonicalRangeError),
}

impl Display for CanonicalAllocationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSize => formatter.write_str("canonical allocation size is invalid"),
            Self::InvalidRange => {
                formatter.write_str("canonical allocation range is out of bounds")
            }
            Self::ResourceExhausted => {
                formatter.write_str("host resources for canonical allocation are exhausted")
            }
            Self::IdentityExhausted(error) => error.fmt(formatter),
            Self::GenerationExhausted(error) => error.fmt(formatter),
            Self::Page(error) => error.fmt(formatter),
            Self::Range(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CanonicalAllocationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Barrier, Mutex};
    use std::thread;

    use crate::{
        DeviceVisibilityPoint, GenerationKind, VisibilityCoordinatorError, VisibilityState,
    };

    #[derive(Default)]
    struct RecordingCoordinator {
        uploads: Mutex<Vec<(DeviceVisibilityRequest, Box<[u8]>)>>,
        downloads: Mutex<Vec<CpuVisibilityRequest>>,
        writeback: Mutex<Box<[u8]>>,
    }

    impl RecordingCoordinator {
        fn with_writeback(bytes: Vec<u8>) -> Self {
            Self {
                writeback: Mutex::new(bytes.into_boxed_slice()),
                ..Self::default()
            }
        }
    }

    impl VisibilityCoordinator for RecordingCoordinator {
        fn make_device_visible(
            &self,
            request: DeviceVisibilityRequest,
            canonical_bytes: &[u8],
        ) -> Result<(), VisibilityCoordinatorError> {
            self.uploads
                .lock()
                .unwrap()
                .push((request, canonical_bytes.into()));
            Ok(())
        }

        fn make_cpu_visible(
            &self,
            request: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            self.downloads.lock().unwrap().push(request);
            Ok(self.writeback.lock().unwrap().clone())
        }
    }

    struct BlockingUploadCoordinator {
        entered: Barrier,
        release: Barrier,
    }

    impl BlockingUploadCoordinator {
        fn new() -> Self {
            Self {
                entered: Barrier::new(2),
                release: Barrier::new(2),
            }
        }
    }

    impl VisibilityCoordinator for BlockingUploadCoordinator {
        fn make_device_visible(
            &self,
            _request: DeviceVisibilityRequest,
            _canonical_bytes: &[u8],
        ) -> Result<(), VisibilityCoordinatorError> {
            self.entered.wait();
            self.release.wait();
            Ok(())
        }

        fn make_cpu_visible(
            &self,
            _request: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            Err(VisibilityCoordinatorError::new(
                "blocking upload coordinator has no device writeback",
            ))
        }
    }

    #[test]
    fn store_epoch_detects_a_mutation_when_the_max_page_generation_does_not_change() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let dominant = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(1),
            0x1000,
            ContentGeneration::new(100),
        )
        .unwrap();
        let changed = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(2),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();

        assert_eq!(store.content_epoch(), ContentMutationEpoch::INITIAL);
        changed
            .write_preflighted(
                0,
                &[0x5a],
                ContentGeneration::INITIAL,
                ContentGeneration::new(1),
            )
            .unwrap();

        assert_eq!(dominant.content_generation(), ContentGeneration::new(100));
        assert_eq!(changed.content_generation(), ContentGeneration::new(1));
        assert_eq!(store.content_epoch(), ContentMutationEpoch::new(1));
    }

    #[test]
    fn rejected_page_write_does_not_advance_the_store_epoch() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let page = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(1),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();

        assert!(matches!(
            page.write_preflighted(
                0,
                &[0x5a],
                ContentGeneration::new(7),
                ContentGeneration::new(8),
            ),
            Err(CanonicalPageError::StaleGeneration { .. })
        ));
        assert_eq!(store.content_epoch(), ContentMutationEpoch::INITIAL);
    }

    #[test]
    fn concurrent_page_publications_do_not_lose_store_epoch_updates() {
        const WRITES_PER_PAGE: u64 = 64;

        let store = CanonicalBackingStore::allocate().unwrap();
        let pages = [
            CanonicalBackingPage::zeroed(
                &store,
                GuestPhysicalPageId::new(1),
                1,
                ContentGeneration::INITIAL,
            )
            .unwrap(),
            CanonicalBackingPage::zeroed(
                &store,
                GuestPhysicalPageId::new(2),
                1,
                ContentGeneration::INITIAL,
            )
            .unwrap(),
        ];
        let workers = pages.map(|page| {
            thread::spawn(move || {
                for generation in 0..WRITES_PER_PAGE {
                    page.write_preflighted(
                        0,
                        &[generation as u8],
                        ContentGeneration::new(generation),
                        ContentGeneration::new(generation + 1),
                    )
                    .unwrap();
                }
            })
        });

        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(
            store.content_epoch(),
            ContentMutationEpoch::new(WRITES_PER_PAGE * 2)
        );
    }

    #[test]
    fn retained_allocation_spans_pages_and_versions_written_contents() {
        let allocation = CanonicalAllocation::zeroed(0x1800, 0x1000).unwrap();
        let before = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        assert_eq!(before.size(), 0x1800);
        assert_eq!(before.segments().len(), 2);
        assert_eq!(before.segments()[1].size(), 0x800);

        allocation.write(0xffe, &[1, 2, 3, 4]).unwrap();
        let mut bytes = [0; 4];
        allocation.read(0xffe, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        assert!(!before.segments()[0].content_is_current());
        assert!(!before.segments()[1].content_is_current());

        let after = allocation.backing_range(MemoryPermissions::READ).unwrap();
        assert_eq!(
            after.segments()[0].content_generation(),
            ContentGeneration::new(1)
        );
        assert_eq!(
            after.segments()[1].content_generation(),
            ContentGeneration::new(1)
        );
        assert_ne!(after.segments()[0].page(), after.segments()[1].page());
        assert_eq!(
            after.segments()[0].page().store(),
            after.segments()[1].page().store()
        );
    }

    #[test]
    fn retained_ranges_outlive_their_allocation_owner() {
        let retained = {
            let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
            allocation.write(7, &[0x5a]).unwrap();
            allocation.backing_range(MemoryPermissions::READ).unwrap()
        };

        assert_eq!(retained.size(), 0x1000);
        assert!(retained.segments()[0].content_is_current());
        assert_eq!(
            retained.segments()[0].content_generation(),
            ContentGeneration::new(1)
        );
    }

    #[test]
    fn device_visibility_round_trip_uses_injected_slow_path() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        allocation.write(4, &[0x11]).unwrap();
        let content_epoch_after_cpu_write = allocation.inner.store.content_epoch();
        let cpu_write_epoch = allocation.inner.store.cpu_write_epoch();
        assert_eq!(cpu_write_epoch, CpuWriteEpoch::new(1));
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::CpuNewer
        );

        let coordinator = Arc::new(RecordingCoordinator::with_writeback(vec![0x5a; 0x1000]));
        let erased: Arc<dyn VisibilityCoordinator> = coordinator.clone();
        let declaration = DeviceAccessDeclaration::read_write(
            NonCpuDeviceId::new(3),
            DeviceVisibilityPoint::new(10),
            DeviceVisibilityPoint::new(11),
        )
        .unwrap();
        range
            .prepare_device_access(declaration, Arc::clone(&erased))
            .unwrap();
        assert_eq!(coordinator.uploads.lock().unwrap().len(), 1);
        assert_eq!(
            coordinator.uploads.lock().unwrap()[0].1[4],
            0x11,
            "the device transition receives current canonical bytes"
        );
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Clean
        );

        range
            .publish_device_write(declaration, Arc::clone(&erased))
            .unwrap();
        let device_write_epoch = allocation.inner.store.content_epoch();
        assert_eq!(
            device_write_epoch,
            content_epoch_after_cpu_write.next().unwrap()
        );
        assert_eq!(
            allocation.inner.store.cpu_write_epoch(),
            cpu_write_epoch,
            "device writes do not invalidate the CPU-write-only fast path"
        );
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::GpuNewer {
                device: NonCpuDeviceId::new(3),
                visible_at: DeviceVisibilityPoint::new(11),
            }
        );

        let before_download = range.segments()[0].content_generation();
        let mut observed = [0; 1];
        allocation.read(4, &mut observed).unwrap();
        assert_eq!(observed, [0x5a]);
        assert_eq!(allocation.inner.store.content_epoch(), device_write_epoch);
        assert_eq!(coordinator.downloads.lock().unwrap().len(), 1);
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Clean
        );
        assert_eq!(
            allocation
                .backing_range(MemoryPermissions::READ)
                .unwrap()
                .segments()[0]
                .content_generation(),
            before_download.next().unwrap()
        );
    }

    #[test]
    fn unsynchronized_devices_produce_a_conflicting_state() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> =
            Arc::new(RecordingCoordinator::with_writeback(vec![0; 0x1000]));
        let first = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(0),
            DeviceVisibilityPoint::new(1),
        )
        .unwrap();
        range
            .prepare_device_access(first, Arc::clone(&coordinator))
            .unwrap();
        range
            .publish_device_write(first, Arc::clone(&coordinator))
            .unwrap();

        let second =
            DeviceAccessDeclaration::read(NonCpuDeviceId::new(2), DeviceVisibilityPoint::new(2));
        assert_eq!(
            range.prepare_device_access(second, coordinator),
            Err(VisibilityError::ConflictingAccess)
        );
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Conflicting
        );
    }

    #[test]
    fn concurrent_cpu_write_rejects_an_in_flight_device_snapshot() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let coordinator = Arc::new(BlockingUploadCoordinator::new());
        let erased: Arc<dyn VisibilityCoordinator> = coordinator.clone();
        let declaration =
            DeviceAccessDeclaration::read(NonCpuDeviceId::new(8), DeviceVisibilityPoint::new(1));
        let worker_range = range.clone();
        let worker = thread::spawn(move || {
            worker_range.prepare_device_access(declaration, Arc::clone(&erased))
        });

        coordinator.entered.wait();
        allocation.write(0, &[0x5a]).unwrap();
        coordinator.release.wait();

        assert_eq!(
            worker.join().unwrap(),
            Err(VisibilityError::ConcurrentTransition)
        );
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Conflicting
        );
        assert!(
            !range.segments()[0].content_is_current(),
            "the concurrent CPU write must publish its content generation"
        );
    }

    #[test]
    fn exhausted_device_writeback_generation_invalidates_without_downloading() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let page = CanonicalBackingPage::initialized(
            &store,
            GuestPhysicalPageId::new(1),
            &[0x11; 0x1000],
            ContentGeneration::MAX,
        )
        .unwrap();
        let range = CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                page.clone(),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                ContentGeneration::MAX,
                MappingGeneration::INITIAL,
            )
            .unwrap(),
        ])
        .unwrap();
        let coordinator = Arc::new(RecordingCoordinator::with_writeback(vec![0x5a; 0x1000]));
        let erased: Arc<dyn VisibilityCoordinator> = coordinator.clone();
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(9),
            DeviceVisibilityPoint::new(2),
            DeviceVisibilityPoint::new(3),
        )
        .unwrap();
        range
            .prepare_device_access(declaration, Arc::clone(&erased))
            .unwrap();
        range
            .publish_device_write(declaration, Arc::clone(&erased))
            .unwrap();

        let mut observed = [0; 1];
        assert_eq!(
            page.read(0, &mut observed),
            Err(CanonicalPageError::Visibility(
                VisibilityError::GenerationExhausted(GenerationExhausted {
                    kind: GenerationKind::Content,
                })
            ))
        );
        assert_eq!(page.visibility_state(), VisibilityState::Invalid);
        assert!(coordinator.downloads.lock().unwrap().is_empty());
    }

    #[test]
    fn canonical_write_batch_commits_cross_page_bytes_once() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let generations = range
            .segments()
            .iter()
            .map(CanonicalBackingSegment::content_generation)
            .collect::<Vec<_>>();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 0x0ffe, &[1, 2, 3, 4]).unwrap();

        let mut before = [0xff; 4];
        allocation.read(0x0ffe, &mut before).unwrap();
        assert_eq!(before, [0; 4]);
        batch.commit().unwrap();
        assert_eq!(
            allocation.inner.store.content_epoch(),
            ContentMutationEpoch::new(1),
            "one failure-atomic batch publishes one store mutation"
        );

        allocation.read(0x0ffe, &mut before).unwrap();
        assert_eq!(before, [1, 2, 3, 4]);
        for (segment, previous) in allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap()
            .segments()
            .iter()
            .zip(generations)
        {
            assert_eq!(segment.content_generation(), previous.next().unwrap());
        }
    }

    #[test]
    fn canonical_write_batch_publishes_once_to_each_distinct_store() {
        let first_store = CanonicalBackingStore::allocate().unwrap();
        let second_store = CanonicalBackingStore::allocate().unwrap();
        let first = CanonicalBackingPage::zeroed(
            &first_store,
            GuestPhysicalPageId::new(1),
            4,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let second = CanonicalBackingPage::zeroed(
            &second_store,
            GuestPhysicalPageId::new(1),
            4,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let range = CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                first,
                0,
                4,
                MemoryPermissions::READ_WRITE,
                ContentGeneration::INITIAL,
                MappingGeneration::INITIAL,
            )
            .unwrap(),
            CanonicalBackingSegment::new(
                second,
                0,
                4,
                MemoryPermissions::READ_WRITE,
                ContentGeneration::INITIAL,
                MappingGeneration::INITIAL,
            )
            .unwrap(),
        ])
        .unwrap();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 2, &[1, 2, 3, 4]).unwrap();

        batch.commit().unwrap();

        assert_eq!(first_store.content_epoch(), ContentMutationEpoch::new(1));
        assert_eq!(second_store.content_epoch(), ContentMutationEpoch::new(1));
    }

    #[test]
    fn canonical_write_batch_rejects_every_page_after_a_concurrent_mutation() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 0x0fff, &[0xaa, 0xbb]).unwrap();
        allocation.write(0x1000, &[0x55]).unwrap();
        let epoch_after_concurrent_write = allocation.inner.store.content_epoch();

        assert_eq!(
            batch.commit(),
            Err(CanonicalWriteBatchError::ConcurrentMutation)
        );
        assert_eq!(
            allocation.inner.store.content_epoch(),
            epoch_after_concurrent_write,
            "a rejected batch must not publish another mutation"
        );
        let mut first = [0xff; 1];
        let mut second = [0xff; 1];
        allocation.read(0x0fff, &mut first).unwrap();
        allocation.read(0x1000, &mut second).unwrap();
        assert_eq!(first, [0]);
        assert_eq!(second, [0x55]);
    }

    #[test]
    fn canonical_write_batch_reads_earlier_staged_bytes_without_publishing_them() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 0x0ffe, &[1, 2, 3, 4]).unwrap();

        let mut staged = [0xff; 6];
        batch.read_staged(&range, 0x0ffd, &mut staged).unwrap();
        assert_eq!(staged, [0, 1, 2, 3, 4, 0]);
        let mut canonical = [0xff; 4];
        allocation.read(0x0ffe, &mut canonical).unwrap();
        assert_eq!(canonical, [0; 4]);
    }

    #[test]
    fn cpu_write_provenance_distinguishes_ranges_within_one_page() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let generation = page.content_generation();

        allocation.write(0x300, &[0x5a]).unwrap();
        assert_eq!(
            page.cpu_write_overlap_since(generation, 0x100, 0x100),
            Ok(CanonicalCpuWriteOverlap::No)
        );
        assert_eq!(
            page.cpu_write_overlap_since(generation, 0x280, 0x100),
            Ok(CanonicalCpuWriteOverlap::Yes)
        );
    }

    #[test]
    fn canonical_write_batch_records_only_staged_dirty_ranges() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let generation = page.content_generation();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 0x700, &[1, 2, 3, 4]).unwrap();
        batch.commit().unwrap();

        assert_eq!(
            page.cpu_write_overlap_since(generation, 0x100, 0x100),
            Ok(CanonicalCpuWriteOverlap::No)
        );
        assert_eq!(
            page.cpu_write_overlap_since(generation, 0x702, 1),
            Ok(CanonicalCpuWriteOverlap::Yes)
        );
    }

    #[test]
    fn expired_cpu_write_provenance_is_conservatively_unknown() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let generation = page.content_generation();
        for index in 0..=CPU_WRITE_JOURNAL_CAPACITY {
            allocation.write(index, &[index as u8]).unwrap();
        }

        assert_eq!(
            page.cpu_write_overlap_since(generation, 0x800, 0x100),
            Ok(CanonicalCpuWriteOverlap::Unknown)
        );
    }

    #[test]
    fn cpu_write_provenance_returns_exact_intervals() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let generation = page.content_generation();
        allocation.write(0x103, &[1, 2, 3]).unwrap();
        allocation.write(0x108, &[4, 5]).unwrap();

        let mut dirty = Vec::new();
        page.append_cpu_write_ranges_since(generation, 0x100, 0x10, &mut dirty)
            .unwrap();
        assert_eq!(
            dirty,
            [
                CanonicalCpuWriteRange::new(0x103, 3),
                CanonicalCpuWriteRange::new(0x108, 2)
            ]
        );
    }

    #[test]
    fn expired_cpu_write_ranges_return_the_complete_requested_interval() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let generation = page.content_generation();
        for index in 0..=CPU_WRITE_JOURNAL_CAPACITY {
            allocation.write(index, &[index as u8]).unwrap();
        }

        let mut dirty = Vec::new();
        page.append_cpu_write_ranges_since(generation, 0x800, 0x100, &mut dirty)
            .unwrap();
        assert_eq!(dirty, [CanonicalCpuWriteRange::new(0x800, 0x100)]);
    }
}
