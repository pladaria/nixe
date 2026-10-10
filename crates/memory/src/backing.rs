//! Retained canonical RAM backing.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::direct::DirectArenaWeak;
use crate::host_mapped::{HOST_BACKING_CAPACITY, HostMappedBacking, HostMappedStore};
use crate::{
    BackingIdentityExhausted, BackingStoreId, CanonicalBackingRange, CanonicalBackingSegment,
    CanonicalPageId, ContentGeneration, CpuVisibilityRequest, DIRECT_PAGE_SIZE,
    DeviceAccessDeclaration, DeviceVisibilityRequest, DirectArena, DirectMemoryError,
    DirectProtectRequest, DirectProtection, ExecutionGate, GenerationExhausted,
    GuestPhysicalPageId, MappingGeneration, MemoryInvalidationKind, MemoryInvalidationLog,
    MemoryInvalidationOrigin, MemoryPermissions, NonCpuDeviceId, VisibilityCoordinator,
    VisibilityError, VisibilityState,
};

struct CanonicalBackingStoreInner {
    identity: BackingStoreId,
    execution_gate: ExecutionGate,
    host: OnceLock<HostMappedStore>,
}

/// Shared authority for every canonical page in one backing store.
///
/// Content generations identify cold canonical/device revisions. Ordinary CPU
/// stores are observed, when needed, through the page dirty epoch instead.
#[derive(Clone)]
pub struct CanonicalBackingStore {
    inner: Arc<CanonicalBackingStoreInner>,
}

impl CanonicalBackingStore {
    /// Allocates a new globally unambiguous backing store.
    pub fn allocate() -> Result<Self, BackingIdentityExhausted> {
        Self::allocate_with_execution_gate(ExecutionGate::new())
    }

    /// Allocates a store governed by an existing process execution gate.
    pub fn allocate_with_execution_gate(
        execution_gate: ExecutionGate,
    ) -> Result<Self, BackingIdentityExhausted> {
        Ok(Self {
            inner: Arc::new(CanonicalBackingStoreInner {
                identity: BackingStoreId::allocate()?,
                execution_gate,
                host: OnceLock::new(),
            }),
        })
    }

    /// Returns the gate that coordinates CPU slices and external transitions.
    #[must_use]
    pub fn execution_gate(&self) -> &ExecutionGate {
        &self.inner.execution_gate
    }

    /// Returns the stable pointer-free store identity.
    #[must_use]
    pub fn identity(&self) -> BackingStoreId {
        self.inner.identity
    }

    fn host(&self) -> Result<&HostMappedStore, CanonicalPageError> {
        if let Some(host) = self.inner.host.get() {
            return Ok(host);
        }
        let host =
            HostMappedStore::new(HOST_BACKING_CAPACITY).map_err(CanonicalPageError::HostMemory)?;
        let _ = self.inner.host.set(host);
        Ok(self
            .inner
            .host
            .get()
            .expect("the host-mapped store was initialized"))
    }
}

impl std::fmt::Debug for CanonicalBackingStore {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CanonicalBackingStore")
            .field("identity", &self.identity())
            .finish()
    }
}

/// Shared completion authority for a homogeneous retained range. Pages retain
/// this record independently of the resource which first published it.
pub(crate) struct RangeDeviceOwner {
    device: NonCpuDeviceId,
    point: AtomicU64,
    fractured: Mutex<bool>,
    coordinator: Arc<dyn VisibilityCoordinator>,
}

impl std::fmt::Debug for RangeDeviceOwner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeDeviceOwner")
            .field("device", &self.device)
            .field("point", &self.point())
            .finish()
    }
}

impl RangeDeviceOwner {
    pub(crate) fn new(
        device: NonCpuDeviceId,
        point: crate::DeviceVisibilityPoint,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Self {
        Self {
            device,
            point: AtomicU64::new(point.get()),
            fractured: Mutex::new(false),
            coordinator,
        }
    }

    pub(crate) fn point(&self) -> crate::DeviceVisibilityPoint {
        crate::DeviceVisibilityPoint::new(self.point.load(Ordering::Acquire))
    }

    pub(crate) fn is_current_for(&self, declaration: DeviceAccessDeclaration) -> bool {
        crate::metrics::record(crate::metrics::Counter::TrackingLocks, 1);
        self.device == declaration.device()
            && self.point() <= declaration.device_visible_at()
            && !*self
                .fractured
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn advance(&self, declaration: DeviceAccessDeclaration) -> bool {
        crate::metrics::record(crate::metrics::Counter::TrackingLocks, 1);
        let fractured = self
            .fractured
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(point) = declaration.cpu_visible_at() else {
            return false;
        };
        // Equal completion points cannot distinguish a racing older readback.
        // Reattach pages in that case so their visibility epochs advance.
        if *fractured || self.device != declaration.device() || self.point() >= point {
            return false;
        }
        self.point.store(point.get(), Ordering::Release);
        true
    }

    pub(crate) fn detach(&self) {
        crate::metrics::record(crate::metrics::Counter::TrackingLocks, 1);
        *self
            .fractured
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
    }

    fn detach_at(&self, expected: crate::DeviceVisibilityPoint) -> bool {
        crate::metrics::record(crate::metrics::Counter::TrackingLocks, 1);
        let mut fractured = self
            .fractured
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.point() != expected {
            return false;
        }
        *fractured = true;
        true
    }
}

enum PageVisibility {
    Clean,
    CpuNewer,
    GpuNewer { owner: Arc<RangeDeviceOwner> },
    Conflicting,
    Invalid,
}

impl PageVisibility {
    fn detach_owner(&self) {
        if let Self::GpuNewer { owner } = self {
            owner.detach();
        }
    }
}

struct CanonicalPageState {
    visibility: PageVisibility,
    visibility_epoch: u64,
    cpu_dirty_observer_armed: bool,
    cpu_dirty_summaries: Vec<(Weak<crate::range::CpuWriteSummary>, usize)>,
    visibility_summaries: Vec<Weak<AtomicU64>>,
    direct_aliases: BTreeMap<(usize, u64), CanonicalDirectAlias>,
}

struct CanonicalDirectAlias {
    arena: DirectArenaWeak,
    guest_address: u64,
    maximum_protection: DirectProtection,
}

struct CanonicalPageInner {
    store: CanonicalBackingStore,
    identity: CanonicalPageId,
    size: usize,
    backing: OnceLock<HostMappedBacking>,
    generation: AtomicU64,
    cpu_dirty_epoch: AtomicU64,
    executable_invalidations: OnceLock<Arc<MemoryInvalidationLog>>,
    // Read-only resident resources query clean authority on every submission.
    // Publish this common state without taking one mutex per physical page;
    // all transitions still serialize through `state`.
    visibility_clean: AtomicBool,
    state: Mutex<CanonicalPageState>,
}

struct CanonicalWriteSnapshot {
    bytes: Box<[u8]>,
    generation: ContentGeneration,
    visibility_epoch: u64,
    dirty_epoch: u64,
}

/// Exact alias permissions gathered while the affected page locks are held.
/// One arena call coalesces equal-permission runs across physical pages.
#[derive(Default)]
struct DirectProtectionBatch {
    arenas: BTreeMap<usize, (DirectArena, BTreeMap<u64, DirectProtectRequest>)>,
}

impl DirectProtectionBatch {
    fn collect(
        &mut self,
        state: &mut CanonicalPageState,
        protection: impl Fn(&CanonicalDirectAlias) -> DirectProtection,
    ) {
        state.direct_aliases.retain(|&(arena_id, _), alias| {
            let Some(arena) = alias.arena.upgrade() else {
                return false;
            };
            self.arenas
                .entry(arena_id)
                .or_insert_with(|| (arena, BTreeMap::new()))
                .1
                .insert(
                    alias.guest_address,
                    DirectProtectRequest {
                        guest_address: alias.guest_address,
                        size: DIRECT_PAGE_SIZE,
                        protection: protection(alias),
                    },
                );
            true
        });
    }

    fn apply(&self) -> Result<(), DirectMemoryError> {
        for (arena, requests) in self.arenas.values() {
            arena.protect_ranges(&requests.values().copied().collect::<Vec<_>>())?;
        }
        Ok(())
    }
}

/// The caller owns ordered execution exclusion for distinct physical pages.
/// No GPU submission, wait or readback occurs while these locks are held.
pub(crate) fn publish_device_pages(
    pages: &[(
        CanonicalBackingPage,
        DeviceAccessDeclaration,
        Arc<RangeDeviceOwner>,
    )],
) -> Result<(), VisibilityError> {
    // Aggregate physical changes, keeping completion-bound resource lifetimes
    // independent, as in Eden's dirty tracking and fence retirement policies.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/67bada77f8a43a90da2e94e89b8e7da73c256989/src/core/gpu_dirty_memory_manager.h
    let logs = executable_write_logs(pages.iter().map(|(page, _, _)| page));
    let reservations = logs
        .values()
        .map(|(log, kinds)| log.reserve_many_from(kinds, MemoryInvalidationOrigin::DeviceWrite))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| VisibilityError::ResourceExhausted)?;
    let mut states = pages
        .iter()
        .map(|(page, _, _)| page.lock_state())
        .collect::<Vec<_>>();
    let mut revoked = DirectProtectionBatch::default();
    for (index, (page, declaration, owner)) in pages.iter().enumerate() {
        match &states[index].visibility {
            PageVisibility::Clean => {
                revoked.collect(&mut states[index], |_| DirectProtection::None)
            }
            PageVisibility::GpuNewer { owner: previous }
                if previous.device == declaration.device() && previous.point() <= owner.point() => {
            }
            PageVisibility::Invalid => return Err(VisibilityError::InvalidState),
            PageVisibility::CpuNewer
            | PageVisibility::GpuNewer { .. }
            | PageVisibility::Conflicting => {
                page.publish_visibility(&mut states[index], PageVisibility::Conflicting)?;
                return Err(VisibilityError::ConflictingAccess);
            }
        }
    }
    revoked
        .apply()
        .map_err(|error| VisibilityError::HostMemory(error.to_string().into()))?;
    for (index, (page, _, owner)) in pages.iter().enumerate() {
        crate::metrics::record_page(
            page.identity(),
            crate::metrics::Counter::DeviceWritePublications,
        );
        page.set_visibility(
            &mut states[index],
            PageVisibility::GpuNewer {
                owner: Arc::clone(owner),
            },
        )?;
    }
    for reservation in reservations {
        reservation.commit();
    }
    Ok(())
}

pub(crate) fn invalidate_device_pages(
    pages: &[CanonicalBackingPage],
) -> Result<(), VisibilityError> {
    let mut states = pages
        .iter()
        .map(CanonicalBackingPage::lock_state)
        .collect::<Vec<_>>();
    let mut revoked = DirectProtectionBatch::default();
    for state in &mut states {
        if !matches!(state.visibility, PageVisibility::GpuNewer { .. }) {
            revoked.collect(state, |_| DirectProtection::None);
        }
    }
    let result = revoked
        .apply()
        .map_err(|error| VisibilityError::HostMemory(error.to_string().into()));
    let mut first_error = result.err();
    for (page, state) in pages.iter().zip(&mut states) {
        if let Err(error) = page.set_visibility(state, PageVisibility::Invalid) {
            first_error.get_or_insert(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

/// The caller owns ordered execution exclusion for these distinct pages.
/// Cache callbacks have the bounded contract of VisibilityCoordinator::cache_cpu_page.
pub(crate) fn prepare_device_pages(
    pages: &[(CanonicalBackingPage, DeviceAccessDeclaration)],
    coordinator: &dyn VisibilityCoordinator,
) -> Result<(), VisibilityError> {
    // Like Eden's changed tracking runs, only real physical changes enter the
    // protection batch; independent resource lifetime references stay intact.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/67bada77f8a43a90da2e94e89b8e7da73c256989/src/video_core/buffer_cache/word_manager.h
    let mut states = pages
        .iter()
        .map(|(page, _)| page.lock_state())
        .collect::<Vec<_>>();
    let mut changed = Vec::new();
    for (index, (page, declaration)) in pages.iter().enumerate() {
        match &states[index].visibility {
            PageVisibility::Clean if !declaration.kind().writes() => {}
            PageVisibility::GpuNewer { owner }
                if owner.device == declaration.device()
                    && owner.point() <= declaration.device_visible_at() => {}
            PageVisibility::Clean | PageVisibility::CpuNewer => changed.push(index),
            PageVisibility::Invalid => return Err(VisibilityError::InvalidState),
            PageVisibility::GpuNewer { .. } | PageVisibility::Conflicting => {
                page.publish_visibility(&mut states[index], PageVisibility::Conflicting)?;
                return Err(VisibilityError::ConflictingAccess);
            }
        }
    }
    if changed.is_empty() {
        return Ok(());
    }
    let result = (|| {
        let mut revoked = DirectProtectionBatch::default();
        for &index in &changed {
            if pages[index].1.kind().writes() {
                revoked.collect(&mut states[index], |_| DirectProtection::None);
            }
        }
        revoked
            .apply()
            .map_err(|error| VisibilityError::HostMemory(error.to_string().into()))?;
        for &index in &changed {
            let (page, declaration) = &pages[index];
            let state = &mut states[index];
            page.inner.visibility_clean.store(false, Ordering::Release);
            if declaration.kind().writes() {
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(page.size())
                    .map_err(|_| VisibilityError::ResourceExhausted)?;
                bytes.resize(page.size(), 0);
                page.load_bytes_quiescent(0, &mut bytes);
                coordinator
                    .cache_cpu_page(
                        DeviceVisibilityRequest {
                            page: page.identity(),
                            size: page.size(),
                            device: declaration.device(),
                            visible_at: declaration.device_visible_at(),
                        },
                        &bytes,
                    )
                    .map_err(VisibilityError::Coordinator)?;
            }
            // Read-only device use never changes canonical bytes. Under this
            // exclusive lease it only needs the final read-only protection;
            // temporary no-access mappings are reserved for device writers.
            crate::metrics::record_page(
                page.identity(),
                crate::metrics::Counter::DeviceReadPreparations,
            );
            page.set_visibility(state, PageVisibility::Clean)?;
        }
        let mut restored = DirectProtectionBatch::default();
        for &index in &changed {
            restored.collect(&mut states[index], |alias| match alias.maximum_protection {
                DirectProtection::ReadWrite => DirectProtection::Read,
                protection => protection,
            });
        }
        restored
            .apply()
            .map_err(|error| VisibilityError::HostMemory(error.to_string().into()))
    })();
    if result.is_err() {
        // No host work has been accepted. Failed protection/cache work leaves
        // the affected pages explicitly invalid, never partly authoritative.
        for &index in &changed {
            let _ = pages[index]
                .0
                .publish_visibility(&mut states[index], PageVisibility::Invalid);
        }
    }
    result
}

/// A read-only consumer can mark protected canonical bytes clean without
/// changing any native alias. The shared execution leases exclude host writes;
/// page locks serialize checked CPU writes and write-fault repair. Return false
/// before changing authority if any page needs protection or device caching.
pub(crate) fn prepare_protected_device_reads(
    pages: &[(CanonicalBackingPage, DeviceAccessDeclaration)],
) -> Result<bool, VisibilityError> {
    if pages
        .iter()
        .any(|(_, declaration)| declaration.kind().writes())
    {
        return Ok(false);
    }
    let mut states = pages
        .iter()
        .map(|(page, _)| page.lock_state())
        .collect::<Vec<_>>();
    for (index, (_, declaration)) in pages.iter().enumerate() {
        match &states[index].visibility {
            PageVisibility::Clean => {}
            PageVisibility::CpuNewer
                if states[index].cpu_dirty_observer_armed
                    && states[index].visibility_epoch != u64::MAX => {}
            PageVisibility::GpuNewer { owner }
                if owner.device == declaration.device()
                    && owner.point() <= declaration.device_visible_at() => {}
            _ => return Ok(false),
        }
    }
    for (index, (page, _)) in pages.iter().enumerate() {
        if matches!(states[index].visibility, PageVisibility::CpuNewer) {
            crate::metrics::record_page(
                page.identity(),
                crate::metrics::Counter::DeviceReadPreparations,
            );
            page.set_visibility(&mut states[index], PageVisibility::Clean)?;
        }
    }
    Ok(true)
}

/// Captures CPU-write epochs and registers one summary for a distinct, ordered
/// page set. `establish_protection` requires exclusive execution admission;
/// otherwise every page must already be protected. Page locks order stores, observer
/// registration and the summary baseline. Select/publish a summary only after
/// locking every covered page, so another capture cannot reuse it before its
/// observers are registered. All captured epochs are sampled before unlock.
pub(crate) fn observe_cpu_pages(
    pages: &[CanonicalBackingPage],
    select_summary: &mut impl FnMut() -> (Arc<crate::range::CpuWriteSummary>, bool),
    establish_protection: bool,
) -> Result<Option<crate::range::CpuWriteObservation>, CanonicalPageError> {
    let mut states = pages
        .iter()
        .map(CanonicalBackingPage::lock_state)
        .collect::<Vec<_>>();
    if establish_protection {
        for (page, state) in pages.iter().zip(&states) {
            if !state.cpu_dirty_observer_armed {
                crate::metrics::record_page(page.identity(), crate::metrics::Counter::ObserverArms);
            }
        }
        arm_cpu_page_states(&mut states)?;
    } else {
        for state in &states {
            match state.visibility {
                PageVisibility::Invalid => {
                    return Err(CanonicalPageError::Visibility(
                        VisibilityError::InvalidState,
                    ));
                }
                PageVisibility::Conflicting => {
                    return Err(CanonicalPageError::Visibility(
                        VisibilityError::ConflictingAccess,
                    ));
                }
                _ => {}
            }
            if !state.cpu_dirty_observer_armed {
                return Ok(None);
            }
        }
    }
    let (summary, shared) = select_summary();
    if !shared {
        for (index, state) in states.iter_mut().enumerate() {
            state
                .cpu_dirty_summaries
                .retain(|(existing, _)| existing.strong_count() != 0);
            state.cpu_dirty_summaries.push((
                Arc::downgrade(&summary),
                index / crate::range::CPU_WRITE_GROUP_PAGES,
            ));
        }
    }
    let epochs = pages
        .iter()
        .map(CanonicalBackingPage::cpu_dirty_epoch)
        .collect();
    Ok(Some(crate::range::CpuWriteObservation::new(
        summary, epochs,
    )))
}

/// Rearms a distinct page set in CanonicalPageId order while its execution
/// gates are held exclusively. Tracking changes need no byte materialization.
pub(crate) fn arm_cpu_pages_quiescent<'a>(
    pages: impl IntoIterator<Item = &'a CanonicalBackingPage>,
) -> Result<Vec<u64>, CanonicalPageError> {
    let pages = pages.into_iter().collect::<Vec<_>>();
    debug_assert!(
        pages
            .windows(2)
            .all(|pair| pair[0].identity() < pair[1].identity())
    );
    let mut states = pages
        .iter()
        .map(|page| page.lock_state())
        .collect::<Vec<_>>();
    for (page, state) in pages.iter().zip(&states) {
        if !state.cpu_dirty_observer_armed {
            crate::metrics::record_page(page.identity(), crate::metrics::Counter::ObserverArms);
        }
    }
    arm_cpu_page_states(&mut states)?;
    Ok(pages.iter().map(|page| page.cpu_dirty_epoch()).collect())
}

fn arm_cpu_page_states(
    states: &mut [std::sync::MutexGuard<'_, CanonicalPageState>],
) -> Result<(), CanonicalPageError> {
    let mut protections = DirectProtectionBatch::default();
    for state in states.iter_mut() {
        match state.visibility {
            PageVisibility::Clean | PageVisibility::CpuNewer | PageVisibility::GpuNewer { .. } => {}
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
        if !state.cpu_dirty_observer_armed {
            let visible = matches!(
                state.visibility,
                PageVisibility::Clean | PageVisibility::CpuNewer
            );
            protections.collect(state, |alias| match (visible, alias.maximum_protection) {
                (false, _) => DirectProtection::None,
                (true, DirectProtection::ReadWrite) => DirectProtection::Read,
                (true, protection) => protection,
            });
        }
    }
    // Keep CPU execution excluded and all affected page states locked until
    // exact alias protections have been coalesced and applied. Publish observer
    // flags only after success; a failed host transition remains an error.
    protections.apply().map_err(|error| {
        CanonicalPageError::Visibility(VisibilityError::HostMemory(error.to_string().into()))
    })?;
    for state in states {
        state.cpu_dirty_observer_armed = true;
    }
    Ok(())
}

/// A retained canonical RAM page.
///
/// Clones retain the bytes independently of CPU mappings and process-owned
/// page tables. Storage and host synchronization details remain private.
#[derive(Clone)]
pub struct CanonicalBackingPage {
    inner: Arc<CanonicalPageInner>,
}

/// Checked RAM access while visibility and dirty tracking cannot change.
/// Callbacks must run after dropping this guard. Multiple pages are locked in
/// CanonicalPageId order, once per physical page (including virtual aliases).
pub struct CanonicalCpuAccess<'a> {
    page: &'a CanonicalBackingPage,
    state: std::sync::MutexGuard<'a, CanonicalPageState>,
}

impl CanonicalCpuAccess<'_> {
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<(), CanonicalPageError> {
        self.page.checked_end(offset, output.len())?;
        self.page.load_bytes_quiescent(offset, output);
        Ok(())
    }

    /// Finish fallible backing/protection work on every affected page before
    /// copying any bytes of a cross-page store. Keep all guards until completion.
    pub fn prepare_write(&mut self) -> Result<(), CanonicalPageError> {
        self.page.ensure_backing()?;
        self.page.prepare_cpu_write_locked(&mut self.state)
    }

    /// Copy after prepare_write, without dropping the page guard between the
    /// dirty transition and these bytes. No generation is added per CPU store.
    pub fn write_prepared(
        &mut self,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), CanonicalPageError> {
        self.page.checked_end(offset, bytes.len())?;
        if self.page.inner.backing.get().is_none() {
            return Err(CanonicalPageError::ResourceExhausted);
        }
        self.page.copy_bytes(offset, bytes);
        Ok(())
    }
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
    /// None requests device reconciliation outside all mapping/page locks.
    /// Successful access never calls a device, takes the execution gate, or
    /// rearms tracking; ordinary CPU writes only perform the first dirty change.
    pub fn try_cpu_access(&self) -> Result<Option<CanonicalCpuAccess<'_>>, CanonicalPageError> {
        let state = self.lock_state();
        if !Self::cpu_visible_locked(&state)? {
            return Ok(None);
        }
        Ok(Some(CanonicalCpuAccess { page: self, state }))
    }
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
                backing: OnceLock::new(),
                generation: AtomicU64::new(generation.get()),
                cpu_dirty_epoch: AtomicU64::new(0),
                executable_invalidations: OnceLock::new(),
                // No device representation exists yet. CPU RAM is writable
                // until an actual consumer arms tracking or takes ownership.
                visibility_clean: AtomicBool::new(false),
                state: Mutex::new(CanonicalPageState {
                    visibility: PageVisibility::CpuNewer,
                    visibility_epoch: 0,
                    cpu_dirty_observer_armed: false,
                    cpu_dirty_summaries: Vec::new(),
                    visibility_summaries: Vec::new(),
                    direct_aliases: BTreeMap::new(),
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
        let backing = allocate_backing(store, bytes.len(), Some(bytes))?;
        let initialized_backing = OnceLock::new();
        initialized_backing
            .set(backing)
            .expect("a new canonical page has no initialized backing");
        Ok(Self {
            inner: Arc::new(CanonicalPageInner {
                store: store.clone(),
                identity: CanonicalPageId::new(store.identity(), page),
                size: bytes.len(),
                backing: initialized_backing,
                generation: AtomicU64::new(generation.get()),
                cpu_dirty_epoch: AtomicU64::new(0),
                executable_invalidations: OnceLock::new(),
                // No device representation exists yet. CPU RAM is writable
                // until an actual consumer arms tracking or takes ownership.
                visibility_clean: AtomicBool::new(false),
                state: Mutex::new(CanonicalPageState {
                    visibility: PageVisibility::CpuNewer,
                    visibility_epoch: 0,
                    cpu_dirty_observer_armed: false,
                    cpu_dirty_summaries: Vec::new(),
                    visibility_summaries: Vec::new(),
                    direct_aliases: BTreeMap::new(),
                }),
            }),
        })
    }

    /// Returns the stable cross-device page identity.
    #[must_use]
    pub fn identity(&self) -> CanonicalPageId {
        self.inner.identity
    }

    /// Permanently connects canonical and device writes on this physical page to
    /// the process-memory invalidation source which owns executable aliases.
    /// Repeating the same subscription is idempotent; a page cannot belong to
    /// two process-memory streams.
    pub fn observe_executable_content(&self, invalidations: Arc<MemoryInvalidationLog>) -> bool {
        if let Some(current) = self.inner.executable_invalidations.get() {
            return Arc::ptr_eq(current, &invalidations);
        }
        match self.inner.executable_invalidations.set(invalidations) {
            Ok(()) => true,
            Err(candidate) => self
                .inner
                .executable_invalidations
                .get()
                .is_some_and(|current| Arc::ptr_eq(current, &candidate)),
        }
    }

    pub(crate) fn store(&self) -> &CanonicalBackingStore {
        &self.inner.store
    }

    /// Returns the byte size of this backing page.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.size
    }

    /// Returns the current byte-content generation.
    #[must_use]
    pub fn content_generation(&self) -> ContentGeneration {
        ContentGeneration::new(self.inner.generation.load(Ordering::Acquire))
    }

    /// Returns the page-granular clean-to-dirty observation epoch.
    #[must_use]
    pub(crate) fn cpu_dirty_epoch(&self) -> u64 {
        self.inner.cpu_dirty_epoch.load(Ordering::Acquire)
    }

    pub(crate) fn needs_cpu_dirty_tracking(&self) -> Result<bool, CanonicalPageError> {
        let state = self.lock_state();
        match state.visibility {
            PageVisibility::Invalid => Err(CanonicalPageError::Visibility(
                VisibilityError::InvalidState,
            )),
            PageVisibility::Conflicting => Err(CanonicalPageError::Visibility(
                VisibilityError::ConflictingAccess,
            )),
            _ => Ok(!state.cpu_dirty_observer_armed),
        }
    }

    pub(crate) fn observe_visibility_summary(&self, summary: &Arc<AtomicU64>) {
        let mut state = self.lock_state();
        state
            .visibility_summaries
            .retain(|existing| existing.strong_count() != 0);
        state.visibility_summaries.push(Arc::downgrade(summary));
    }

    fn notify_visibility_summaries(state: &mut CanonicalPageState) {
        state.visibility_summaries.retain(|summary| {
            if let Some(summary) = summary.upgrade() {
                let _ = summary.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                    Some(value.saturating_add(1))
                });
                true
            } else {
                false
            }
        });
    }

    /// Materializes and retains the canonical shared-file view used by direct
    /// guest-address aliases. This does not grant CPU access or change page
    /// visibility.
    pub fn direct_backing(&self) -> Result<HostMappedBacking, CanonicalPageError> {
        self.ensure_backing()?;
        Ok(self
            .inner
            .backing
            .get()
            .expect("materialized canonical backing is retained")
            .clone())
    }

    /// Registers one derived virtual alias for physical-page-wide revocation.
    pub fn register_direct_alias(
        &self,
        arena: &DirectArena,
        guest_address: u64,
        maximum_protection: DirectProtection,
    ) -> Result<(), DirectMemoryError> {
        if !guest_address.is_multiple_of(DIRECT_PAGE_SIZE as u64) {
            return Err(DirectMemoryError::invalid_contract(
                "direct alias guest address is not page aligned",
            ));
        }
        let mut state = self.lock_state();
        crate::metrics::record_page(
            self.identity(),
            crate::metrics::Counter::DirectAliasRegistrations,
        );
        state.direct_aliases.insert(
            (arena.identity(), guest_address),
            CanonicalDirectAlias {
                arena: arena.downgrade(),
                guest_address,
                maximum_protection,
            },
        );
        self.publish_direct_alias_protection(&mut state)
    }

    /// Removes one derived alias after its host mapping has been revoked.
    pub fn unregister_direct_alias(&self, arena: &DirectArena, guest_address: u64) {
        self.lock_state()
            .direct_aliases
            .remove(&(arena.identity(), guest_address));
    }

    /// Resolves the cold first-write transition before retrying the native
    /// store which faulted. The page becomes CPU-visible and dirty before any
    /// writable alias is republished.
    pub fn resolve_direct_write_fault(&self) -> Result<bool, CanonicalPageError> {
        #[cfg(feature = "performance-counters")]
        {
            use crate::metrics::Counter;
            let state = self.lock_state();
            let counter = match (&state.visibility, state.cpu_dirty_observer_armed) {
                (PageVisibility::Clean, false) => Counter::WriteFaultCleanUnobserved,
                (PageVisibility::Clean, true) => Counter::WriteFaultCleanObserved,
                (PageVisibility::CpuNewer, false) => Counter::WriteFaultCpuUnobserved,
                (PageVisibility::CpuNewer, true) => Counter::WriteFaultCpuObserved,
                (PageVisibility::GpuNewer { .. }, _) => Counter::WriteFaultDeviceOwned,
                _ => Counter::WriteFaultInvalid,
            };
            crate::metrics::record_page(self.identity(), counter);
        }
        self.prepare_cpu_write()?;
        Ok(true)
    }

    /// Returns the conservative authority state shared by every page alias.
    #[must_use]
    pub fn visibility_state(&self) -> VisibilityState {
        if self.inner.visibility_clean.load(Ordering::Acquire) {
            VisibilityState::Clean
        } else {
            Self::visibility_snapshot(&self.lock_state().visibility)
        }
    }

    /// Copies a checked byte range out of canonical storage.
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<(), CanonicalPageError> {
        self.checked_end(offset, output.len())?;
        loop {
            let state = self.lock_state();
            match state.visibility {
                PageVisibility::Clean | PageVisibility::CpuNewer => {
                    self.load_bytes_quiescent(offset, output);
                    return Ok(());
                }
                PageVisibility::GpuNewer { .. } => {
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

    /// Atomically reads one naturally aligned scalar from canonical storage.
    pub fn atomic_load(&self, offset: usize, size: usize) -> Result<u128, CanonicalPageError> {
        loop {
            if let Some(value) = self.try_atomic_load(offset, size)? {
                return Ok(value);
            }
            self.prepare_cpu_access()?;
        }
    }

    /// Performs the same atomic load without invoking a visibility callback.
    /// Native callers resolve None only after releasing execution admission.
    pub fn try_atomic_load(
        &self,
        offset: usize,
        size: usize,
    ) -> Result<Option<u128>, CanonicalPageError> {
        self.checked_end(offset, size)?;
        self.ensure_backing()?;
        let state = self.lock_state();
        if !Self::cpu_visible_locked(&state)? {
            return Ok(None);
        }
        self.inner
            .backing
            .get()
            .expect("atomic access materialized canonical backing")
            .atomic_load(offset, size)
            .map(Some)
            .map_err(|error| {
                CanonicalPageError::Visibility(VisibilityError::HostMemory(
                    error.to_string().into_boxed_str(),
                ))
            })
    }

    /// Atomically compares and conditionally replaces one naturally aligned
    /// scalar in canonical storage. Every physical alias shares this location.
    pub fn atomic_compare_exchange(
        &self,
        offset: usize,
        size: usize,
        expected: u128,
        replacement: u128,
    ) -> Result<(u128, bool), CanonicalPageError> {
        self.checked_end(offset, size)?;
        self.ensure_backing()?;
        loop {
            let mut state = self.lock_state();
            if !Self::cpu_visible_locked(&state)? {
                drop(state);
                self.prepare_cpu_access()?;
                continue;
            }
            // Keep dirty publication and the hardware CAS indivisible with
            // tracking rearm/snapshots. Native atomics use their execution lease
            // instead; this lock is only on the checked/cold backing path.
            self.prepare_cpu_write_locked(&mut state)?;
            return self
                .inner
                .backing
                .get()
                .expect("atomic access materialized canonical backing")
                .atomic_compare_exchange(offset, size, expected, replacement)
                .map_err(|error| {
                    CanonicalPageError::Visibility(VisibilityError::HostMemory(
                        error.to_string().into_boxed_str(),
                    ))
                });
        }
    }

    /// Copies an already protected CPU-visible page under shared execution
    /// admission. The page lock orders checked writers and fault repair;
    /// native stores still fault. False requests normal visibility/exclusion.
    pub(crate) fn read_protected(
        &self,
        offset: usize,
        output: &mut [u8],
    ) -> Result<bool, CanonicalPageError> {
        self.checked_end(offset, output.len())?;
        let state = self.lock_state();
        if !state.cpu_dirty_observer_armed || !Self::cpu_visible_locked(&state)? {
            return Ok(false);
        }
        self.load_bytes_quiescent(offset, output);
        Ok(true)
    }

    /// Copies bytes while the caller holds this store's execution gate
    /// exclusively. The gate excludes native writers; the page lock also
    /// orders retained checked atomics with the byte copy.
    pub(crate) fn read_quiescent(
        &self,
        offset: usize,
        output: &mut [u8],
    ) -> Result<(), CanonicalPageError> {
        self.checked_end(offset, output.len())?;
        let state = self.lock_state();
        match state.visibility {
            PageVisibility::Clean | PageVisibility::CpuNewer => {
                self.load_bytes_quiescent(offset, output);
                Ok(())
            }
            PageVisibility::GpuNewer { .. } => Err(CanonicalPageError::Visibility(
                VisibilityError::ConcurrentTransition,
            )),
            PageVisibility::Conflicting => Err(CanonicalPageError::Visibility(
                VisibilityError::ConflictingAccess,
            )),
            PageVisibility::Invalid => Err(CanonicalPageError::Visibility(
                VisibilityError::InvalidState,
            )),
        }
    }

    /// Reports whether canonical bytes may be copied while the caller holds
    /// this store's execution gate exclusively.
    pub(crate) fn cpu_visible_quiescent(&self) -> Result<bool, CanonicalPageError> {
        Self::cpu_visible_locked(&self.lock_state())
    }

    fn cpu_visible_locked(state: &CanonicalPageState) -> Result<bool, CanonicalPageError> {
        match state.visibility {
            PageVisibility::Clean | PageVisibility::CpuNewer => Ok(true),
            PageVisibility::GpuNewer { .. } => Ok(false),
            PageVisibility::Conflicting => Err(CanonicalPageError::Visibility(
                VisibilityError::ConflictingAccess,
            )),
            PageVisibility::Invalid => Err(CanonicalPageError::Visibility(
                VisibilityError::InvalidState,
            )),
        }
    }

    /// Captures one page under its state lock and a memory execution lease.
    /// Arming native write protection requires an exclusive lease. With a
    /// shared lease, an already armed page is stable under this lock: native
    /// stores still fault, checked stores take the same lock, and host writers
    /// require execution exclusion. None requests the exclusive path.
    fn snapshot_cpu_write(
        &self,
        arm_observer: bool,
    ) -> Result<Option<CanonicalWriteSnapshot>, CanonicalPageError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.size())
            .map_err(|_| CanonicalPageError::ResourceExhausted)?;
        bytes.resize(self.size(), 0);
        let mut state = self.lock_state();
        match state.visibility {
            PageVisibility::Clean | PageVisibility::CpuNewer => {}
            PageVisibility::GpuNewer { .. } if !arm_observer => return Ok(None),
            PageVisibility::GpuNewer { .. } => {
                return Err(CanonicalPageError::Visibility(
                    VisibilityError::ConcurrentTransition,
                ));
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
        if !state.cpu_dirty_observer_armed {
            if !arm_observer {
                return Ok(None);
            }
            crate::metrics::record_page(self.identity(), crate::metrics::Counter::ObserverArms);
            state.cpu_dirty_observer_armed = true;
            if let Err(error) = self.publish_direct_alias_protection(&mut state) {
                state.cpu_dirty_observer_armed = false;
                return Err(CanonicalPageError::Visibility(VisibilityError::HostMemory(
                    error.to_string().into_boxed_str(),
                )));
            }
        }
        self.load_bytes_quiescent(0, &mut bytes);
        let generation = self.content_generation();
        let visibility_epoch = state.visibility_epoch;
        let dirty_epoch = self.cpu_dirty_epoch();
        Ok(Some(CanonicalWriteSnapshot {
            bytes: bytes.into_boxed_slice(),
            generation,
            visibility_epoch,
            dirty_epoch,
        }))
    }

    /// Materializes zero storage before a multi-page write is published.
    ///
    /// Allocation may fail, but a successful call does not change guest bytes
    /// or their content generation.
    pub fn prepare_write(&self) -> Result<(), CanonicalPageError> {
        self.prepare_cpu_access()?;
        self.ensure_backing()?;
        let mut state = self.lock_state();
        self.require_cpu_authority(&mut state)
            .map_err(CanonicalPageError::Visibility)?;
        Ok(())
    }

    /// Prepares an ordinary CPU write without creating a per-store version.
    pub fn prepare_cpu_write(&self) -> Result<(), CanonicalPageError> {
        self.prepare_cpu_access()?;
        self.ensure_backing()?;
        let mut state = self.lock_state();
        self.prepare_cpu_write_locked(&mut state)
    }

    fn prepare_cpu_write_locked(
        &self,
        state: &mut CanonicalPageState,
    ) -> Result<(), CanonicalPageError> {
        self.require_cpu_authority(state)
            .map_err(CanonicalPageError::Visibility)?;
        if matches!(state.visibility, PageVisibility::Clean) {
            self.publish_visibility(state, PageVisibility::CpuNewer)
                .map_err(CanonicalPageError::Visibility)?;
        }
        self.publish_cpu_dirty(state)
    }

    /// Establishes canonical CPU visibility without reading or modifying bytes.
    ///
    /// CPU adapters use this before operations which require canonical bytes
    /// to have been reconciled from a device owner.
    pub fn prepare_cpu_access(&self) -> Result<(), CanonicalPageError> {
        self.ensure_cpu_visible()
            .map_err(CanonicalPageError::Visibility)
    }

    /// Atomically writes bytes and publishes a preflighted next generation.
    ///
    /// The expected generation prevents two cold generation-publishing
    /// transactions from committing the same revision. Ordinary CPU writes
    /// are deliberately outside this generation protocol.
    pub fn write_preflighted(
        &self,
        offset: usize,
        bytes: &[u8],
        expected: ContentGeneration,
        next: ContentGeneration,
    ) -> Result<(), CanonicalPageError> {
        self.checked_end(offset, bytes.len())?;
        self.ensure_backing()?;
        if expected.next() != Ok(next) {
            return Err(CanonicalPageError::InvalidGenerationTransition);
        }
        let mut state = self.lock_state();
        self.require_cpu_authority(&mut state)
            .map_err(CanonicalPageError::Visibility)?;
        if matches!(state.visibility, PageVisibility::Clean) {
            self.publish_visibility(&mut state, PageVisibility::CpuNewer)
                .map_err(CanonicalPageError::Visibility)?;
        }
        self.publish_cpu_dirty(&mut state)?;
        let observed = self.content_generation();
        if observed != expected {
            return Err(CanonicalPageError::StaleGeneration { expected, observed });
        }
        self.copy_bytes(offset, bytes);
        self.inner.generation.store(next.get(), Ordering::Release);
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
        self.checked_end(offset, bytes.len())?;
        self.ensure_backing()?;
        let mut state = self.lock_state();
        self.require_cpu_authority(&mut state)
            .map_err(CanonicalPageError::Visibility)?;
        if matches!(state.visibility, PageVisibility::Clean) {
            self.publish_visibility(&mut state, PageVisibility::CpuNewer)
                .map_err(CanonicalPageError::Visibility)?;
        }
        self.publish_cpu_dirty(&mut state)?;
        let observed = self.content_generation();
        if observed != generation {
            return Err(CanonicalPageError::StaleGeneration {
                expected: generation,
                observed,
            });
        }
        self.copy_bytes(offset, bytes);
        Ok(())
    }

    pub(crate) fn attach_resident_device_owner(
        &self,
        declaration: DeviceAccessDeclaration,
        owner: &Arc<RangeDeviceOwner>,
    ) -> Result<bool, VisibilityError> {
        let mut state = self.lock_state();
        if !matches!(&state.visibility, PageVisibility::GpuNewer { owner: previous }
            if previous.device == declaration.device() && previous.point() <= owner.point())
        {
            return Ok(false);
        }
        self.publish_visibility(
            &mut state,
            PageVisibility::GpuNewer {
                owner: Arc::clone(owner),
            },
        )?;
        Ok(true)
    }

    fn ensure_cpu_visible(&self) -> Result<(), VisibilityError> {
        self.ensure_cpu_visible_with(&mut |coordinator, request| {
            coordinator.make_cpu_visible(request)
        })
    }

    pub(crate) fn ensure_cpu_visible_with(
        &self,
        resolve: &mut crate::CpuVisibilityResolver<'_>,
    ) -> Result<(), VisibilityError> {
        loop {
            let (owner, visible_at, epoch, next_generation) = {
                let mut state = self.lock_state();
                match &state.visibility {
                    PageVisibility::Clean | PageVisibility::CpuNewer => return Ok(()),
                    PageVisibility::Conflicting => {
                        return Err(VisibilityError::ConflictingAccess);
                    }
                    PageVisibility::Invalid => return Err(VisibilityError::InvalidState),
                    PageVisibility::GpuNewer { owner } => {
                        let next = match self.content_generation().next() {
                            Ok(next) => next,
                            Err(error) => {
                                self.publish_visibility(&mut state, PageVisibility::Invalid)?;
                                return Err(VisibilityError::GenerationExhausted(error));
                            }
                        };
                        (
                            Arc::clone(owner),
                            owner.point(),
                            state.visibility_epoch,
                            next,
                        )
                    }
                }
            };
            self.ensure_backing()
                .map_err(|_| VisibilityError::ResourceExhausted)?;
            let request = CpuVisibilityRequest {
                page: self.identity(),
                size: self.size(),
                device: owner.device,
                visible_at,
            };
            let writeback = resolve(owner.coordinator.as_ref(), request);
            let mut state = self.lock_state();
            if state.visibility_epoch != epoch {
                match state.visibility {
                    PageVisibility::Clean | PageVisibility::CpuNewer => return Ok(()),
                    PageVisibility::GpuNewer { .. } => continue,
                    PageVisibility::Conflicting => {
                        return Err(VisibilityError::ConflictingAccess);
                    }
                    PageVisibility::Invalid => return Err(VisibilityError::InvalidState),
                }
            }
            // A range publication advances one shared point, without changing
            // this page's epoch. Serialize detachment against that advance so
            // a completed older readback can never expose newer GPU bytes.
            if !owner.detach_at(visible_at) {
                continue;
            }
            let bytes = match writeback {
                Ok(bytes) => bytes,
                Err(error) => {
                    self.publish_visibility(&mut state, PageVisibility::Invalid)?;
                    return Err(VisibilityError::Coordinator(error));
                }
            };
            if bytes.len() != self.size() {
                let observed = bytes.len();
                self.publish_visibility(&mut state, PageVisibility::Invalid)?;
                return Err(VisibilityError::IncorrectWritebackSize {
                    expected: self.size(),
                    observed,
                });
            }

            // Keep every direct alias inaccessible until canonical bytes and
            // their cold revision have been committed. The final visibility
            // publication is the only operation which may reopen host access.
            self.store_bytes(0, &bytes);
            self.inner
                .generation
                .store(next_generation.get(), Ordering::Release);
            self.publish_cpu_dirty(&mut state)
                .map_err(|error| match error {
                    CanonicalPageError::CpuDirtyEpochExhausted => {
                        VisibilityError::CpuDirtyEpochExhausted
                    }
                    CanonicalPageError::Visibility(error) => error,
                    _ => unreachable!("CPU dirty publication has no other failure mode"),
                })?;
            crate::metrics::record_page(self.identity(), crate::metrics::Counter::CpuReadbacks);
            self.publish_visibility(&mut state, PageVisibility::Clean)?;
            return Ok(());
        }
    }

    fn visibility_snapshot(visibility: &PageVisibility) -> VisibilityState {
        match visibility {
            PageVisibility::Clean => VisibilityState::Clean,
            PageVisibility::CpuNewer => VisibilityState::CpuNewer,
            PageVisibility::GpuNewer { owner } => VisibilityState::GpuNewer {
                device: owner.device,
                visible_at: owner.point(),
            },
            PageVisibility::Conflicting => VisibilityState::Conflicting,
            PageVisibility::Invalid => VisibilityState::Invalid,
        }
    }

    fn require_cpu_authority(&self, state: &mut CanonicalPageState) -> Result<(), VisibilityError> {
        match state.visibility {
            PageVisibility::Clean | PageVisibility::CpuNewer => Ok(()),
            PageVisibility::GpuNewer { .. } | PageVisibility::Conflicting => {
                self.publish_visibility(state, PageVisibility::Conflicting)?;
                Err(VisibilityError::ConflictingAccess)
            }
            PageVisibility::Invalid => Err(VisibilityError::InvalidState),
        }
    }

    fn publish_visibility(
        &self,
        state: &mut CanonicalPageState,
        visibility: PageVisibility,
    ) -> Result<(), VisibilityError> {
        let cpu_visible = matches!(visibility, PageVisibility::Clean | PageVisibility::CpuNewer);
        let cpu_writable = matches!(visibility, PageVisibility::CpuNewer);
        let already_revoked = matches!(state.visibility, PageVisibility::GpuNewer { .. });
        if !(already_revoked && !cpu_visible)
            && let Err(error) =
                self.publish_direct_alias_protection_for(state, cpu_visible, cpu_writable)
        {
            self.set_visibility(state, PageVisibility::Invalid)?;
            return Err(VisibilityError::HostMemory(
                error.to_string().into_boxed_str(),
            ));
        }
        self.set_visibility(state, visibility)
    }

    fn set_visibility(
        &self,
        state: &mut CanonicalPageState,
        visibility: PageVisibility,
    ) -> Result<(), VisibilityError> {
        Self::notify_visibility_summaries(state);
        if !matches!((&state.visibility, &visibility),
            (PageVisibility::GpuNewer { owner: old }, PageVisibility::GpuNewer { owner: new })
                if Arc::ptr_eq(old, new))
        {
            state.visibility.detach_owner();
        }
        self.inner.visibility_clean.store(false, Ordering::Release);
        let Some(next_epoch) = state.visibility_epoch.checked_add(1) else {
            state.visibility.detach_owner();
            state.visibility = PageVisibility::Invalid;
            return Err(VisibilityError::VisibilityEpochExhausted);
        };
        state.visibility = visibility;
        state.visibility_epoch = next_epoch;
        self.inner.visibility_clean.store(
            matches!(state.visibility, PageVisibility::Clean),
            Ordering::Release,
        );
        Ok(())
    }

    fn publish_cpu_dirty(&self, state: &mut CanonicalPageState) -> Result<(), CanonicalPageError> {
        if !state.cpu_dirty_observer_armed {
            return Ok(());
        }
        self.advance_cpu_dirty_epoch(state)?;
        state.cpu_dirty_observer_armed = false;
        self.publish_direct_alias_protection(state)
            .map_err(|error| {
                CanonicalPageError::Visibility(VisibilityError::HostMemory(
                    error.to_string().into_boxed_str(),
                ))
            })?;
        Ok(())
    }

    // Host mutations do not retry a native store. They invalidate observers
    // while retaining read-only aliases, so subsequent snapshots need not
    // unprotect and immediately protect the same CPU page again. A real CPU
    // write still uses publish_cpu_dirty to reopen its writable aliases.
    fn advance_cpu_dirty_epoch(
        &self,
        state: &mut CanonicalPageState,
    ) -> Result<(), CanonicalPageError> {
        if !state.cpu_dirty_observer_armed {
            return Ok(());
        }
        let Some(next) = self.cpu_dirty_epoch().checked_add(1) else {
            self.revoke_direct_access(state)
                .map_err(CanonicalPageError::Visibility)?;
            self.inner.visibility_clean.store(false, Ordering::Release);
            state.visibility.detach_owner();
            Self::notify_visibility_summaries(state);
            state.visibility = PageVisibility::Invalid;
            return Err(CanonicalPageError::CpuDirtyEpochExhausted);
        };
        self.inner.cpu_dirty_epoch.store(next, Ordering::Release);
        state.cpu_dirty_summaries.retain(|(summary, group)| {
            if let Some(summary) = summary.upgrade() {
                summary.publish(*group);
                true
            } else {
                false
            }
        });
        Ok(())
    }

    fn revoke_direct_access(&self, state: &mut CanonicalPageState) -> Result<(), VisibilityError> {
        let protection = if matches!(state.visibility, PageVisibility::GpuNewer { .. }) {
            Ok(())
        } else {
            self.publish_direct_aliases_as(state, DirectProtection::None)
        };
        protection.map_err(|error| {
            self.inner.visibility_clean.store(false, Ordering::Release);
            state.visibility.detach_owner();
            Self::notify_visibility_summaries(state);
            state.visibility = PageVisibility::Invalid;
            VisibilityError::HostMemory(error.to_string().into_boxed_str())
        })?;
        self.inner.visibility_clean.store(false, Ordering::Release);
        let Some(next_epoch) = state.visibility_epoch.checked_add(1) else {
            self.inner.visibility_clean.store(false, Ordering::Release);
            state.visibility.detach_owner();
            Self::notify_visibility_summaries(state);
            state.visibility = PageVisibility::Invalid;
            return Err(VisibilityError::VisibilityEpochExhausted);
        };
        state.visibility_epoch = next_epoch;
        Ok(())
    }

    fn publish_direct_alias_protection(
        &self,
        state: &mut CanonicalPageState,
    ) -> Result<(), DirectMemoryError> {
        self.publish_direct_alias_protection_for(
            state,
            matches!(
                state.visibility,
                PageVisibility::Clean | PageVisibility::CpuNewer
            ),
            matches!(state.visibility, PageVisibility::CpuNewer),
        )
    }

    fn publish_direct_alias_protection_for(
        &self,
        state: &mut CanonicalPageState,
        cpu_visible: bool,
        cpu_writable: bool,
    ) -> Result<(), DirectMemoryError> {
        let write_enabled = cpu_visible && cpu_writable && !state.cpu_dirty_observer_armed;
        self.publish_direct_aliases(state, |alias| {
            match (cpu_visible, alias.maximum_protection) {
                (false, _) => DirectProtection::None,
                (true, DirectProtection::ReadWrite) if !write_enabled => DirectProtection::Read,
                (true, protection) => protection,
            }
        })
    }

    fn publish_direct_aliases_as(
        &self,
        state: &mut CanonicalPageState,
        protection: DirectProtection,
    ) -> Result<(), DirectMemoryError> {
        self.publish_direct_aliases(state, |_| protection)
    }

    fn publish_direct_aliases(
        &self,
        state: &mut CanonicalPageState,
        protection: impl Fn(&CanonicalDirectAlias) -> DirectProtection,
    ) -> Result<(), DirectMemoryError> {
        let mut arenas = BTreeMap::<usize, (DirectArena, Vec<DirectProtectRequest>)>::new();
        state.direct_aliases.retain(|&(arena_id, _), alias| {
            let Some(arena) = alias.arena.upgrade() else {
                return false;
            };
            arenas
                .entry(arena_id)
                .or_insert_with(|| (arena, Vec::new()))
                .1
                .push(DirectProtectRequest {
                    guest_address: alias.guest_address,
                    size: DIRECT_PAGE_SIZE,
                    protection: protection(alias),
                });
            true
        });
        for (arena, requests) in arenas.values_mut() {
            requests.sort_unstable_by_key(|request| request.guest_address);
            arena.protect_ranges(requests)?;
        }
        Ok(())
    }

    fn checked_end(&self, offset: usize, size: usize) -> Result<usize, CanonicalPageError> {
        offset
            .checked_add(size)
            .filter(|end| *end <= self.inner.size)
            .ok_or(CanonicalPageError::InvalidRange)
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, CanonicalPageState> {
        crate::metrics::record(crate::metrics::Counter::TrackingLocks, 1);
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn load_bytes_quiescent(&self, offset: usize, output: &mut [u8]) {
        let Some(backing) = self.inner.backing.get() else {
            output.fill(0);
            return;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                (backing.base() + offset) as *const u8,
                output.as_mut_ptr(),
                output.len(),
            );
        }
    }

    fn store_bytes(&self, offset: usize, bytes: &[u8]) {
        self.copy_bytes(offset, bytes);
    }

    fn copy_bytes(&self, offset: usize, bytes: &[u8]) {
        let backing = self
            .inner
            .backing
            .get()
            .expect("canonical writes materialize host backing during preflight");
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (backing.base() + offset) as *mut u8,
                bytes.len(),
            );
        }
    }

    fn ensure_backing(&self) -> Result<(), CanonicalPageError> {
        if self.inner.backing.get().is_some() {
            return Ok(());
        }
        let backing = allocate_backing(self.store(), self.size(), None)?;
        let _ = self.inner.backing.set(backing);
        Ok(())
    }
}

fn allocate_backing(
    store: &CanonicalBackingStore,
    size: usize,
    contents: Option<&[u8]>,
) -> Result<HostMappedBacking, CanonicalPageError> {
    store
        .host()?
        .allocate(size, contents)
        .map_err(CanonicalPageError::HostMemory)
}

fn executable_write_logs<'a>(
    pages: impl Iterator<Item = &'a CanonicalBackingPage>,
) -> BTreeMap<usize, (Arc<MemoryInvalidationLog>, Vec<MemoryInvalidationKind>)> {
    let mut logs = BTreeMap::new();
    for page in pages {
        if let Some(log) = page.inner.executable_invalidations.get() {
            let (_, kinds) = logs
                .entry(Arc::as_ptr(log).addr())
                .or_insert_with(|| (log.clone(), Vec::new()));
            kinds.push(MemoryInvalidationKind::ExecutableContent {
                first: page.identity().page(),
                second: None,
            });
        }
    }
    logs
}

struct PendingCanonicalPageWrite {
    backing: CanonicalBackingPage,
    expected_generation: ContentGeneration,
    expected_visibility_epoch: u64,
    expected_dirty_epoch: u64,
    bytes: Box<[u8]>,
    dirty_ranges: Vec<(u64, u64)>,
}

impl PendingCanonicalPageWrite {
    fn new(backing: CanonicalBackingPage, snapshot: CanonicalWriteSnapshot) -> Self {
        Self {
            backing,
            expected_generation: snapshot.generation,
            expected_visibility_epoch: snapshot.visibility_epoch,
            expected_dirty_epoch: snapshot.dirty_epoch,
            bytes: snapshot.bytes,
            dirty_ranges: Vec::new(),
        }
    }

    fn record_dirty_range(&mut self, start: u64, end: u64) {
        if let Some((previous_start, previous_end)) = self.dirty_ranges.last_mut()
            && start <= *previous_end
            && end >= *previous_start
        {
            *previous_start = (*previous_start).min(start);
            *previous_end = (*previous_end).max(end);
        } else {
            self.dirty_ranges.push((start, end));
        }
    }
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

    fn prepare_unstaged_cpu_visible(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        end: u64,
    ) -> Result<(), CanonicalWriteBatchError> {
        let mut visited = BTreeSet::new();
        for (logical_start, segment) in range.segments_between(offset, end) {
            let logical_end = logical_start
                .checked_add(segment.size())
                .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
            if offset.max(logical_start) < end.min(logical_end)
                && !self.pages.contains_key(&segment.page())
                && visited.insert(segment.page())
            {
                segment
                    .backing()
                    .prepare_cpu_access()
                    .map_err(CanonicalWriteBatchError::Page)?;
            }
        }
        Ok(())
    }

    fn unstaged_cpu_visible_quiescent(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        end: u64,
    ) -> Result<bool, CanonicalWriteBatchError> {
        let mut visited = BTreeSet::new();
        for (logical_start, segment) in range.segments_between(offset, end) {
            let logical_end = logical_start
                .checked_add(segment.size())
                .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
            if offset.max(logical_start) < end.min(logical_end)
                && !self.pages.contains_key(&segment.page())
                && visited.insert(segment.page())
                && !segment
                    .backing()
                    .cpu_visible_quiescent()
                    .map_err(CanonicalWriteBatchError::Page)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Returns whether this transaction modifies any byte in one logical
    /// canonical subrange.
    pub fn overlaps(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        size: u64,
    ) -> Result<bool, CanonicalWriteBatchError> {
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
        if size == 0 || self.pages.is_empty() {
            return Ok(false);
        }

        for (logical_start, segment) in range.segments_between(offset, end) {
            let logical_end = logical_start
                .checked_add(segment.size())
                .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
            let read_start = offset.max(logical_start);
            let read_end = end.min(logical_end);
            if read_start < read_end
                && let Some(pending) = self.pages.get(&segment.page())
            {
                let page_start = segment
                    .offset()
                    .checked_add(read_start - logical_start)
                    .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
                let page_end = page_start
                    .checked_add(read_end - read_start)
                    .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
                if pending
                    .dirty_ranges
                    .iter()
                    .any(|&(start, end)| start < page_end && page_start < end)
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
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
        loop {
            self.prepare_unstaged_cpu_visible(range, offset, end)?;
            let mut stores = BTreeMap::new();
            for (_, segment) in range.segments_between(offset, end) {
                // Captured pages are private, immutable snapshots. Reading them
                // neither observes canonical bytes nor needs a CPU rendezvous.
                if !self.pages.contains_key(&segment.page()) {
                    stores
                        .entry(segment.backing().store().execution_gate().identity())
                        .or_insert_with(|| segment.backing().store().execution_gate().clone());
                }
            }
            let _transitions = stores
                .values()
                .map(ExecutionGate::acquire_exclusive)
                .collect::<Vec<_>>();
            if !self.unstaged_cpu_visible_quiescent(range, offset, end)? {
                continue;
            }

            let mut copied = 0_usize;
            for (logical_start, segment) in range.segments_between(offset, end) {
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
                            .read_quiescent(page_offset, &mut output[copied..copied_end])
                            .map_err(CanonicalWriteBatchError::Page)?;
                    }
                    copied = copied_end;
                }
            }
            if copied != output.len() {
                return Err(CanonicalWriteBatchError::IncompleteRange);
            }
            return Ok(());
        }
    }

    /// Reads current canonical bytes with only this batch's modified ranges
    /// overlaid. Device command streams must not hide a later CPU write to
    /// unrelated bytes merely because both lie in the same captured page.
    pub fn read_overlay(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        output: &mut [u8],
    ) -> Result<(), CanonicalWriteBatchError> {
        let size = output.len() as u64;
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
        let covered = range
            .segments_between(offset, end)
            .all(|(logical, segment)| {
                let start = offset.max(logical);
                let stop = end.min(logical + segment.size());
                let page_start = segment.offset() + start - logical;
                let page_end = page_start + stop - start;
                self.pages.get(&segment.page()).is_some_and(|pending| {
                    pending
                        .dirty_ranges
                        .iter()
                        .any(|&(start, end)| start <= page_start && end >= page_end)
                })
            });
        if covered {
            return self.read_staged(range, offset, output);
        }
        range
            .read(offset, output)
            .map_err(|error| CanonicalWriteBatchError::Read(Box::new(error)))?;
        for (logical, segment) in range.segments_between(offset, end) {
            let Some(pending) = self.pages.get(&segment.page()) else {
                continue;
            };
            let start = offset.max(logical);
            let stop = end.min(logical + segment.size());
            let page_start = segment.offset() + start - logical;
            let page_end = page_start + stop - start;
            for &(dirty_start, dirty_end) in &pending.dirty_ranges {
                let patch_start = page_start.max(dirty_start);
                let patch_end = page_end.min(dirty_end);
                if patch_start < patch_end {
                    let out_start = (start - offset + patch_start - page_start) as usize;
                    let count = (patch_end - patch_start) as usize;
                    output[out_start..out_start + count]
                        .copy_from_slice(&pending.bytes[patch_start as usize..patch_end as usize]);
                }
            }
        }
        Ok(())
    }

    /// Stages one checked logical write. Discard the batch if this fails.
    /// The first snapshot of a physical page arms tracking under memory
    /// exclusion. Call without an execution lease or cache lock.
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

        for (_, segment) in range.segments_between(offset, end) {
            if !segment.permissions().contains(MemoryPermissions::WRITE) {
                return Err(CanonicalWriteBatchError::PermissionDenied {
                    page: segment.page(),
                    available: segment.permissions(),
                });
            }
        }

        loop {
            self.prepare_unstaged_cpu_visible(range, offset, end)?;

            // Only newly captured pages touch canonical storage or arm tracking.
            // Existing staged pages are owned bytes, even if their backing has
            // since changed. Acquire distinct gates in stable identity order, with
            // no page/log mutex held while waiting for execution to stop.
            let mut fresh_pages = BTreeMap::new();
            for (logical_start, segment) in range.segments_between(offset, end) {
                let logical_end = logical_start
                    .checked_add(segment.size())
                    .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
                if offset.max(logical_start) < end.min(logical_end)
                    && !self.pages.contains_key(&segment.page())
                {
                    fresh_pages
                        .entry(segment.page())
                        .or_insert_with(|| segment.backing());
                }
            }
            // One protected physical page needs no native rendezvous merely
            // to copy its bytes. Multi-page snapshots retain exclusive admission
            // so their combined observation remains atomic across CPU workers.
            if fresh_pages.len() == 1 {
                let (&id, &page) = fresh_pages.first_key_value().expect("one fresh page");
                let _lease = page.store().execution_gate().acquire_shared();
                if let Some(snapshot) = page
                    .snapshot_cpu_write(false)
                    .map_err(CanonicalWriteBatchError::Page)?
                {
                    self.pages
                        .insert(id, PendingCanonicalPageWrite::new(page.clone(), snapshot));
                }
            }
            let mut stores = BTreeMap::new();
            for (id, page) in fresh_pages {
                if !self.pages.contains_key(&id) {
                    let gate = page.store().execution_gate();
                    stores
                        .entry(gate.identity())
                        .or_insert_with(|| gate.clone());
                }
            }
            let _transitions = stores
                .values()
                .map(ExecutionGate::acquire_exclusive)
                .collect::<Vec<_>>();
            if !self.unstaged_cpu_visible_quiescent(range, offset, end)? {
                continue;
            }
            // Staging changes neither virtual mappings nor byte generations;
            // retain the memory mapping epoch.

            let mut copied = 0_usize;
            for (logical_start, segment) in range.segments_between(offset, end) {
                let logical_end = logical_start
                    .checked_add(segment.size())
                    .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
                let write_start = offset.max(logical_start);
                let write_end = end.min(logical_end);
                if write_start < write_end {
                    let pending = match self.pages.entry(segment.page()) {
                        std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            let snapshot = segment
                                .backing()
                                .snapshot_cpu_write(true)
                                .map_err(CanonicalWriteBatchError::Page)?
                                .expect("exclusive admission may arm the write observer");
                            entry.insert(PendingCanonicalPageWrite::new(
                                segment.backing().clone(),
                                snapshot,
                            ))
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
                    pending.record_dirty_range(page_offset as u64, (page_offset + count) as u64);
                    copied = copied_end;
                }
            }
            if copied != bytes.len() {
                return Err(CanonicalWriteBatchError::IncompleteRange);
            }
            return Ok(());
        }
    }

    /// Publishes every staged byte mutation and advances each page once.
    pub fn commit(self) -> Result<(), CanonicalWriteBatchError> {
        self.commit_inner(false)
    }

    /// Publishes ordered device writes onto the latest CPU-visible page.
    /// Only modified ranges are copied: concurrent CPU writes to neighboring
    /// bytes survive, and an overlapping device write follows command order.
    /// Unlike a checked host transaction, this has no stale-page rollback.
    pub fn commit_ordered(self) -> Result<(), CanonicalWriteBatchError> {
        self.commit_inner(true)
    }

    fn commit_inner(self, ordered: bool) -> Result<(), CanonicalWriteBatchError> {
        struct Write {
            expected_generation: ContentGeneration,
            expected_visibility_epoch: u64,
            expected_dirty_epoch: u64,
            next_generation: ContentGeneration,
            next_visibility_epoch: u64,
            bytes: Option<Box<[u8]>>,
            dirty_ranges: Vec<(u64, u64)>,
        }

        let mut stores = BTreeMap::new();
        for pending in self.pages.values() {
            stores
                .entry(pending.backing.store().execution_gate().identity())
                .or_insert_with(|| pending.backing.store().execution_gate().clone());
        }
        let _transitions = stores
            .values()
            .map(ExecutionGate::acquire_exclusive)
            .collect::<Vec<_>>();

        // Reserve each stream once, in stable order, only after the engine has
        // drained. A reservation owns the log mutex and cannot span rendezvous.
        let logs = executable_write_logs(self.pages.values().map(|pending| &pending.backing));
        let reservations = logs
            .values()
            .map(|(log, kinds)| log.reserve_many_from(kinds, MemoryInvalidationOrigin::HostWrite))
            .collect::<Result<Vec<_>, _>>()
            .map_err(CanonicalWriteBatchError::Invalidation)?;

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
                expected_dirty_epoch: pending.expected_dirty_epoch,
                next_generation,
                next_visibility_epoch,
                bytes: Some(pending.bytes),
                dirty_ranges: pending.dirty_ranges,
            });
        }
        for backing in &backings {
            backing
                .ensure_backing()
                .map_err(CanonicalWriteBatchError::Page)?;
        }

        let mut states = backings
            .iter()
            .map(CanonicalBackingPage::lock_state)
            .collect::<Vec<_>>();
        for ((backing, state), write) in backings.iter().zip(&states).zip(&mut writes) {
            if (!ordered
                && (backing.content_generation() != write.expected_generation
                    || state.visibility_epoch != write.expected_visibility_epoch
                    || backing.cpu_dirty_epoch() != write.expected_dirty_epoch))
                || !matches!(
                    state.visibility,
                    PageVisibility::Clean | PageVisibility::CpuNewer
                )
            {
                return Err(CanonicalWriteBatchError::ConcurrentMutation);
            }
            write.next_generation = backing
                .content_generation()
                .next()
                .map_err(CanonicalWriteBatchError::GenerationExhausted)?;
            write.next_visibility_epoch = state
                .visibility_epoch
                .checked_add(1)
                .ok_or(CanonicalWriteBatchError::VisibilityEpochExhausted)?;
            if state.cpu_dirty_observer_armed && backing.cpu_dirty_epoch() == u64::MAX {
                return Err(CanonicalWriteBatchError::Page(
                    CanonicalPageError::CpuDirtyEpochExhausted,
                ));
            }
        }
        for ((backing, state), write) in backings.iter().zip(states.iter_mut()).zip(&mut writes) {
            let bytes = write
                .bytes
                .take()
                .expect("prepared canonical batch write retains its bytes");
            backing
                .inner
                .visibility_clean
                .store(false, Ordering::Release);
            state.visibility.detach_owner();
            CanonicalBackingPage::notify_visibility_summaries(state);
            state.visibility = PageVisibility::CpuNewer;
            backing
                .advance_cpu_dirty_epoch(state)
                .expect("CPU dirty epochs were preflighted while page states are locked");
            if ordered {
                for &(start, end) in &write.dirty_ranges {
                    backing.store_bytes(start as usize, &bytes[start as usize..end as usize]);
                }
            } else {
                backing.store_bytes(0, &bytes);
            }
            backing
                .inner
                .generation
                .store(write.next_generation.get(), Ordering::Release);
            state.visibility_epoch = write.next_visibility_epoch;
        }
        drop(states);
        for reservation in reservations {
            reservation.commit();
        }
        Ok(())
    }
}

/// Failure while staging or atomically committing canonical writes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalWriteBatchError {
    Read(Box<crate::CanonicalRangeAccessError>),
    Invalidation(crate::MemoryInvalidationError),
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
            Self::Read(error) => error.fmt(formatter),
            Self::Invalidation(error) => error.fmt(formatter),
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
    /// Canonical metadata allocation failed.
    ResourceExhausted,
    /// Host backing allocation or mapping failed with its original cause.
    HostMemory(crate::HostMappedError),
    /// The supplied generation was no longer current.
    StaleGeneration {
        expected: ContentGeneration,
        observed: ContentGeneration,
    },
    /// The supplied next generation was not the exact successor.
    InvalidGenerationTransition,
    /// The page-granular CPU dirty epoch cannot advance without wrapping.
    CpuDirtyEpochExhausted,
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
            Self::HostMemory(error) => error.fmt(formatter),
            Self::StaleGeneration { expected, observed } => write!(
                formatter,
                "canonical page generation changed: expected {expected}, observed {observed}"
            ),
            Self::InvalidGenerationTransition => {
                formatter.write_str("canonical page generation transition is not consecutive")
            }
            Self::CpuDirtyEpochExhausted => {
                formatter.write_str("canonical CPU dirty epoch is exhausted")
            }
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
}

/// Retained canonical allocation shared by kernel objects, CPU aliases and
/// device mappings.
#[derive(Clone, Debug)]
pub struct CanonicalAllocation {
    inner: Arc<CanonicalAllocationInner>,
}

impl CanonicalAllocation {
    /// Creates a zero-filled allocation divided into fixed-size canonical pages.
    pub fn zeroed(size: usize, page_size: usize) -> Result<Self, CanonicalAllocationError> {
        let store = CanonicalBackingStore::allocate()
            .map_err(CanonicalAllocationError::IdentityExhausted)?;
        Self::zeroed_in(store, GuestPhysicalPageId::new(1), size, page_size)
    }

    /// Allocates pages in an existing ownership domain. The owner must reserve
    /// these physical identities before calling this method.
    pub fn zeroed_in(
        store: CanonicalBackingStore,
        first_page: GuestPhysicalPageId,
        size: usize,
        page_size: usize,
    ) -> Result<Self, CanonicalAllocationError> {
        if size == 0 || page_size == 0 || !page_size.is_power_of_two() {
            return Err(CanonicalAllocationError::InvalidSize);
        }
        let page_count = size.div_ceil(page_size);
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(page_count)
            .map_err(|_| CanonicalAllocationError::ResourceExhausted)?;
        for index in 0..page_count {
            let local_id = u64::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(first_page.get()))
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

    /// Canonical pages retained by both kernel objects and CPU mappings.
    #[must_use]
    pub fn pages(&self) -> &[CanonicalBackingPage] {
        &self.inner.pages
    }

    /// Copies a checked logical range out of canonical backing.
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<(), CanonicalAllocationError> {
        let end = self.checked_end(offset, output.len())?;
        if output.is_empty() {
            return Ok(());
        }
        let first_page = offset / self.inner.page_size;
        let last_page = (end - 1) / self.inner.page_size;
        loop {
            for page in &self.inner.pages[first_page..=last_page] {
                page.prepare_cpu_access()
                    .map_err(CanonicalAllocationError::Page)?;
            }
            let _execution = self.inner.store.execution_gate().acquire_exclusive();
            let cpu_visible = self.inner.pages[first_page..=last_page]
                .iter()
                .map(CanonicalBackingPage::cpu_visible_quiescent)
                .collect::<Result<Vec<_>, _>>()
                .map_err(CanonicalAllocationError::Page)?;
            if !cpu_visible.into_iter().all(std::convert::identity) {
                continue;
            }
            let mut cursor = offset;
            let mut copied = 0;
            while cursor < end {
                let page_index = cursor / self.inner.page_size;
                let page_offset = cursor % self.inner.page_size;
                let count = (self.inner.page_size - page_offset).min(end - cursor);
                self.inner.pages[page_index]
                    .read_quiescent(page_offset, &mut output[copied..copied + count])
                    .map_err(CanonicalAllocationError::Page)?;
                cursor += count;
                copied += count;
            }
            return Ok(());
        }
    }

    /// Atomically writes a checked logical range and advances each affected
    /// page generation once.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), CanonicalAllocationError> {
        let end = self.checked_end(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        let first_page = offset / self.inner.page_size;
        let last_page = (end - 1) / self.inner.page_size;
        if first_page == last_page {
            // Kernel producers predominantly update short records. Keep this
            // route allocation-free for non-executable data, while retaining
            // the same rendezvous, tracking and executable invalidations.
            let page = &self.inner.pages[first_page];
            loop {
                page.prepare_write()
                    .map_err(CanonicalAllocationError::Page)?;
                let _execution = self.inner.store.execution_gate().acquire_exclusive();
                if !page
                    .cpu_visible_quiescent()
                    .map_err(CanonicalAllocationError::Page)?
                {
                    continue;
                }
                let changes = [MemoryInvalidationKind::ExecutableContent {
                    first: page.identity().page(),
                    second: None,
                }];
                let reservation = page
                    .inner
                    .executable_invalidations
                    .get()
                    .map(|log| log.reserve_many_from(&changes, MemoryInvalidationOrigin::HostWrite))
                    .transpose()
                    .map_err(CanonicalAllocationError::Invalidation)?;
                let generation = page.content_generation();
                let next = generation
                    .next()
                    .map_err(CanonicalAllocationError::GenerationExhausted)?;
                page.write_preflighted(offset % self.inner.page_size, bytes, generation, next)
                    .map_err(CanonicalAllocationError::Page)?;
                if let Some(reservation) = reservation {
                    reservation.commit();
                }
                return Ok(());
            }
        }
        for page in &self.inner.pages[first_page..=last_page] {
            page.prepare_write()
                .map_err(CanonicalAllocationError::Page)?;
        }
        let _execution = loop {
            let execution = self.inner.store.execution_gate().acquire_exclusive();
            if self.inner.pages[first_page..=last_page]
                .iter()
                .map(CanonicalBackingPage::cpu_visible_quiescent)
                .collect::<Result<Vec<_>, _>>()
                .map_err(CanonicalAllocationError::Page)?
                .into_iter()
                .all(std::convert::identity)
            {
                break execution;
            }
            drop(execution);
            for page in &self.inner.pages[first_page..=last_page] {
                page.prepare_write()
                    .map_err(CanonicalAllocationError::Page)?;
            }
        };
        // The gate owns byte serialization, including retained-range writers.
        // Reserve only after rendezvous, and publish before reopening admission.
        let logs = executable_write_logs(self.inner.pages[first_page..=last_page].iter());
        let reservations = logs
            .values()
            .map(|(log, kinds)| log.reserve_many_from(kinds, MemoryInvalidationOrigin::HostWrite))
            .collect::<Result<Vec<_>, _>>()
            .map_err(CanonicalAllocationError::Invalidation)?;
        let mut generations = Vec::new();
        generations
            .try_reserve_exact(last_page - first_page + 1)
            .map_err(|_| CanonicalAllocationError::ResourceExhausted)?;
        for page in &self.inner.pages[first_page..=last_page] {
            let current = page.content_generation();
            let next = current
                .next()
                .map_err(CanonicalAllocationError::GenerationExhausted)?;
            generations.push(next);
        }
        let mut states = self.inner.pages[first_page..=last_page]
            .iter()
            .map(CanonicalBackingPage::lock_state)
            .collect::<Vec<_>>();
        // Finish fallible protection/dirty work for every page before copying
        // any bytes. No full-page snapshot or rollback buffer is needed.
        for (page, state) in self.inner.pages[first_page..=last_page]
            .iter()
            .zip(&mut states)
        {
            page.require_cpu_authority(state).map_err(|error| {
                CanonicalAllocationError::Page(CanonicalPageError::Visibility(error))
            })?;
            if matches!(state.visibility, PageVisibility::Clean) {
                page.publish_visibility(state, PageVisibility::CpuNewer)
                    .map_err(|error| {
                        CanonicalAllocationError::Page(CanonicalPageError::Visibility(error))
                    })?;
            }
            page.publish_cpu_dirty(state)
                .map_err(CanonicalAllocationError::Page)?;
        }
        let mut cursor = offset;
        let mut copied = 0;
        while cursor < end {
            let page_index = cursor / self.inner.page_size;
            let page_offset = cursor % self.inner.page_size;
            let count = (self.inner.page_size - page_offset).min(end - cursor);
            let page = &self.inner.pages[page_index];
            page.store_bytes(page_offset, &bytes[copied..copied + count]);
            page.inner.generation.store(
                generations[page_index - first_page].get(),
                Ordering::Release,
            );
            cursor += count;
            copied += count;
        }
        drop(states);
        for reservation in reservations {
            reservation.commit();
        }
        Ok(())
    }

    /// Creates a retained pointer-free view over the complete allocation.
    pub fn backing_range(
        &self,
        permissions: MemoryPermissions,
    ) -> Result<CanonicalBackingRange, CanonicalAllocationError> {
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
}

/// Failure while creating or accessing a retained canonical allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalAllocationError {
    Invalidation(crate::MemoryInvalidationError),
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
            Self::Invalidation(error) => error.fmt(formatter),
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
        CanonicalCpuWriteDependency, DeviceVisibilityPoint, DirectArena, DirectMapRequest,
        DirectProtection, GenerationKind, VisibilityCoordinatorError, VisibilityState,
    };

    #[test]
    fn ordered_device_overlay_preserves_later_cpu_bytes_outside_its_patch() {
        let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        allocation.write(0, &[0x11; 16]).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut writes = CanonicalWriteBatch::new();
        writes.stage(&range, 1, &[0xa1, 0xb2]).unwrap();
        allocation.write(0, &[0x22; 16]).unwrap();
        let mut captured = [0; 4];
        writes.read_staged(&range, 0, &mut captured).unwrap();
        assert_eq!(captured, [0x11, 0xa1, 0xb2, 0x11]);
        let mut ordered = [0; 16];
        writes.read_overlay(&range, 0, &mut ordered).unwrap();
        let mut expected = [0x22; 16];
        expected[1..3].copy_from_slice(&[0xa1, 0xb2]);
        assert_eq!(ordered, expected);
        writes.commit_ordered().unwrap();
        allocation.read(0, &mut ordered).unwrap();
        assert_eq!(ordered, expected);
    }

    #[test]
    fn retained_device_range_prevents_offset_reuse_and_new_pages_keep_distinct_identity() {
        let store = CanonicalBackingStore::allocate().unwrap();
        store
            .inner
            .host
            .set(HostMappedStore::new(DIRECT_PAGE_SIZE).unwrap())
            .unwrap();
        let first = CanonicalBackingPage::initialized(
            &store,
            GuestPhysicalPageId::new(1),
            &[0x5a; DIRECT_PAGE_SIZE],
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let pointer = first.direct_backing().unwrap().base();
        let first_identity = first.identity();
        let retained = CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                first.clone(),
                0,
                DIRECT_PAGE_SIZE as u64,
                MemoryPermissions::READ_WRITE,
                MappingGeneration::INITIAL,
            )
            .unwrap(),
        ])
        .unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> =
            Arc::new(RecordingCoordinator::with_writeback(vec![
                0xa5;
                DIRECT_PAGE_SIZE
            ]));
        let write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(0),
            DeviceVisibilityPoint::new(1),
        )
        .unwrap();
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&retained, write)],
            coordinator.clone(),
        )
        .unwrap();
        CanonicalBackingRange::publish_device_writes([(&retained, write)], coordinator).unwrap();
        drop(first);
        let second = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(2),
            DIRECT_PAGE_SIZE,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        assert_ne!(second.identity(), first_identity);
        assert!(matches!(
            second.direct_backing(),
            Err(CanonicalPageError::HostMemory(_))
        ));
        let mut observed = [0; 1];
        retained.read(0, &mut observed).unwrap();
        assert_eq!(observed, [0xa5]);
        assert!(second.direct_backing().is_err());
        drop(retained);
        let reused = second.direct_backing().unwrap();
        assert_eq!(reused.base(), pointer);
        second.read(0, &mut observed).unwrap();
        assert_eq!(observed, [0]);
    }

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
        fn cache_cpu_page(
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

    struct BlockingDownloadCoordinator {
        entered: Barrier,
        release: Barrier,
        downloads: std::sync::atomic::AtomicUsize,
        writeback: Box<[u8]>,
    }

    impl BlockingDownloadCoordinator {
        fn new(writeback: Vec<u8>) -> Self {
            Self {
                entered: Barrier::new(3),
                release: Barrier::new(3),
                downloads: std::sync::atomic::AtomicUsize::new(0),
                writeback: writeback.into_boxed_slice(),
            }
        }
    }

    impl VisibilityCoordinator for BlockingDownloadCoordinator {
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
            self.downloads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.entered.wait();
            self.release.wait();
            Ok(self.writeback.clone())
        }
    }

    #[test]
    fn cpu_store_between_preparation_and_publication_is_not_overwritten() {
        let memory = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = memory.backing_range(MemoryPermissions::READ_WRITE).unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(RecordingCoordinator::default());
        let write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            crate::DeviceVisibilityPoint::new(1),
            crate::DeviceVisibilityPoint::new(1),
        )
        .unwrap();
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, write)],
            coordinator.clone(),
        )
        .unwrap();
        memory.write(0x1000, &[0x75]).unwrap();
        assert_eq!(
            CanonicalBackingRange::publish_device_writes([(&range, write)], coordinator),
            Err(VisibilityError::ConflictingAccess)
        );
        let mut byte = [0];
        memory.pages()[1].load_bytes_quiescent(0, &mut byte);
        assert_eq!(byte, [0x75]);
        assert!(!memory.inner.store.execution_gate().transition_pending());
    }

    #[test]
    fn partial_publication_failure_keeps_new_owner_and_releases_exclusion() {
        let memory = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = memory.backing_range(MemoryPermissions::READ_WRITE).unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(RecordingCoordinator::default());
        let write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            crate::DeviceVisibilityPoint::new(1),
            crate::DeviceVisibilityPoint::new(1),
        )
        .unwrap();
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, write)],
            coordinator.clone(),
        )
        .unwrap();
        memory.pages()[1].lock_state().visibility_epoch = u64::MAX;
        assert_eq!(
            CanonicalBackingRange::publish_device_writes([(&range, write)], coordinator),
            Err(VisibilityError::VisibilityEpochExhausted)
        );
        assert_eq!(
            memory.pages()[0].visibility_state(),
            VisibilityState::GpuNewer {
                device: write.device(),
                visible_at: write.cpu_visible_at().unwrap()
            }
        );
        assert_eq!(
            memory.pages()[1].visibility_state(),
            VisibilityState::Invalid
        );
        assert!(!memory.inner.store.execution_gate().transition_pending());
        assert_eq!(
            CanonicalBackingRange::invalidate_visibility_ranges([&range]),
            Err(VisibilityError::VisibilityEpochExhausted)
        );
        assert!(
            memory
                .pages()
                .iter()
                .all(|page| page.visibility_state() == VisibilityState::Invalid)
        );
        assert!(!memory.inner.store.execution_gate().transition_pending());
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
        let cpu_writes = CanonicalCpuWriteDependency::capture(&before).unwrap();

        allocation.write(0xffe, &[1, 2, 3, 4]).unwrap();
        let mut bytes = [0; 4];
        allocation.read(0xffe, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        assert!(!cpu_writes.remains_current());

        let after = allocation.backing_range(MemoryPermissions::READ).unwrap();
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
        let mut observed = [0; 1];
        retained.read(7, &mut observed).unwrap();
        assert_eq!(observed, [0x5a]);
    }

    #[test]
    fn device_visibility_round_trip_uses_injected_slow_path() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        allocation.write(4, &[0x11]).unwrap();
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
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, declaration)],
            Arc::clone(&erased),
        )
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

        crate::CanonicalBackingRange::publish_device_writes(
            [(&range, declaration)],
            Arc::clone(&erased),
        )
        .unwrap();
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::GpuNewer {
                device: NonCpuDeviceId::new(3),
                visible_at: DeviceVisibilityPoint::new(11),
            }
        );

        let before_download = range.segments()[0].backing().content_generation();
        let mut observed = [0; 1];
        allocation.read(4, &mut observed).unwrap();
        assert_eq!(observed, [0x5a]);
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
                .backing()
                .content_generation(),
            before_download.next().unwrap()
        );
    }

    #[test]
    fn device_newer_write_fault_reconciles_then_dirties_in_one_page_resolver() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let host = page.direct_backing().unwrap();
        let arena = DirectArena::new(0x3000).unwrap();
        arena
            .map_pages(&[DirectMapRequest {
                guest_address: 0x1000,
                backing: &host,
                protection: DirectProtection::Read,
            }])
            .unwrap();
        page.register_direct_alias(&arena, 0x1000, DirectProtection::ReadWrite)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let mut device_bytes = vec![0; 0x1000];
        device_bytes[7] = 0xa5;
        let coordinator: Arc<dyn VisibilityCoordinator> =
            Arc::new(RecordingCoordinator::with_writeback(device_bytes));
        let declaration = DeviceAccessDeclaration::read_write(
            NonCpuDeviceId::new(9),
            DeviceVisibilityPoint::new(30),
            DeviceVisibilityPoint::new(31),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, declaration)],
            Arc::clone(&coordinator),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, declaration)], coordinator)
            .unwrap();
        assert_eq!(arena.protection_at(0x1000), Some(DirectProtection::None));

        assert!(page.resolve_direct_write_fault().unwrap());

        let mut observed = [0];
        page.read(7, &mut observed).unwrap();
        assert_eq!(observed, [0xa5]);
        assert_eq!(page.visibility_state(), VisibilityState::CpuNewer);
        assert_eq!(
            arena.protection_at(0x1000),
            Some(DirectProtection::ReadWrite)
        );
        assert!(!dependency.remains_current());
    }

    #[test]
    fn concurrent_device_writeback_reconciliation_commits_one_page_revision() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let host = page.direct_backing().unwrap();
        let arena = DirectArena::new(0x3000).unwrap();
        arena
            .map_pages(&[DirectMapRequest {
                guest_address: 0x1000,
                backing: &host,
                protection: DirectProtection::Read,
            }])
            .unwrap();
        page.register_direct_alias(&arena, 0x1000, DirectProtection::ReadWrite)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let generation = page.content_generation();
        let coordinator = Arc::new(BlockingDownloadCoordinator::new(vec![0xa5; 0x1000]));
        let erased: Arc<dyn VisibilityCoordinator> = coordinator.clone();
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(11),
            DeviceVisibilityPoint::new(50),
            DeviceVisibilityPoint::new(51),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, declaration)],
            Arc::clone(&erased),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, declaration)], erased)
            .unwrap();

        let workers = (0..2)
            .map(|_| {
                let page = page.clone();
                thread::spawn(move || page.prepare_cpu_access())
            })
            .collect::<Vec<_>>();
        coordinator.entered.wait();
        assert_eq!(arena.protection_at(0x1000), Some(DirectProtection::None));
        coordinator.release.wait();

        for worker in workers {
            worker.join().unwrap().unwrap();
        }
        assert_eq!(
            coordinator
                .downloads
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        assert_eq!(page.content_generation(), generation.next().unwrap());
        assert_eq!(page.visibility_state(), VisibilityState::Clean);
        assert_eq!(arena.protection_at(0x1000), Some(DirectProtection::Read));
        assert!(!dependency.remains_current());
        let mut observed = [0; 1];
        page.read(0, &mut observed).unwrap();
        assert_eq!(observed, [0xa5]);
    }

    #[test]
    fn device_materialization_publishes_page_dirty_without_a_cpu_fault() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> =
            Arc::new(RecordingCoordinator::with_writeback(vec![0x5a; 0x1000]));
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(10),
            DeviceVisibilityPoint::new(40),
            DeviceVisibilityPoint::new(41),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, declaration)],
            Arc::clone(&coordinator),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, declaration)], coordinator)
            .unwrap();

        assert_eq!(page.cpu_dirty_epoch(), 0);
        assert!(dependency.remains_current());
        let mut observed = [0; 1];
        range.read(0, &mut observed).unwrap();
        assert_eq!(observed, [0x5a]);
        assert_eq!(page.cpu_dirty_epoch(), 1);
        assert!(!dependency.remains_current());
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
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, first)],
            Arc::clone(&coordinator),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes(
            [(&range, first)],
            Arc::clone(&coordinator),
        )
        .unwrap();

        let second =
            DeviceAccessDeclaration::read(NonCpuDeviceId::new(2), DeviceVisibilityPoint::new(2));
        assert_eq!(
            crate::CanonicalBackingRange::prepare_resident_device_accesses(
                [(&range, second)],
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
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, declaration)],
            Arc::clone(&erased),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes(
            [(&range, declaration)],
            Arc::clone(&erased),
        )
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
            .map(|segment| segment.backing().content_generation())
            .collect::<Vec<_>>();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 0x0ffe, &[1, 2, 3, 4]).unwrap();

        let mut before = [0xff; 4];
        allocation.read(0x0ffe, &mut before).unwrap();
        assert_eq!(before, [0; 4]);
        batch.commit().unwrap();

        allocation.read(0x0ffe, &mut before).unwrap();
        assert_eq!(before, [1, 2, 3, 4]);
        for (segment, previous) in allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap()
            .segments()
            .iter()
            .zip(generations)
        {
            assert_eq!(
                segment.backing().content_generation(),
                previous.next().unwrap()
            );
        }
    }

    #[test]
    fn canonical_write_batch_preserves_direct_read_watch_across_staging_and_commit() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let host = page.direct_backing().unwrap();
        let arena = DirectArena::new(0x4000).unwrap();
        arena
            .map_pages(&[DirectMapRequest {
                guest_address: 0x1000,
                backing: &host,
                protection: DirectProtection::Read,
            }])
            .unwrap();
        page.register_direct_alias(&arena, 0x1000, DirectProtection::ReadWrite)
            .unwrap();

        let mut abandoned = CanonicalWriteBatch::new();
        abandoned.stage(&range, 0, &[0x11]).unwrap();
        assert_eq!(arena.protection_at(0x1000), Some(DirectProtection::Read));
        drop(abandoned);
        assert_eq!(arena.protection_at(0x1000), Some(DirectProtection::Read));

        let mut committed = CanonicalWriteBatch::new();
        committed.stage(&range, 0, &[0x22]).unwrap();
        committed.commit().unwrap();
        assert_eq!(arena.protection_at(0x1000), Some(DirectProtection::Read));
    }

    #[test]
    fn cpu_dirty_capture_protects_every_physical_alias() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let host = page.direct_backing().unwrap();
        let first = DirectArena::new(0x4000).unwrap();
        let second = DirectArena::new(0x4000).unwrap();
        for (arena, address) in [(&first, 0x1000), (&second, 0x2000)] {
            arena
                .map_pages(&[DirectMapRequest {
                    guest_address: address,
                    backing: &host,
                    protection: DirectProtection::Read,
                }])
                .unwrap();
            page.register_direct_alias(arena, address, DirectProtection::ReadWrite)
                .unwrap();
        }
        allocation.write(0, &[1]).unwrap();
        assert_eq!(
            first.protection_at(0x1000),
            Some(DirectProtection::ReadWrite)
        );
        assert_eq!(
            second.protection_at(0x2000),
            Some(DirectProtection::ReadWrite)
        );

        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();

        assert!(dependency.remains_current());
        assert_eq!(first.protection_at(0x1000), Some(DirectProtection::Read));
        assert_eq!(second.protection_at(0x2000), Some(DirectProtection::Read));
        allocation.write(0x800, &[2]).unwrap();
        assert!(!dependency.remains_current());
    }

    #[test]
    fn concurrent_first_write_transitions_advance_one_dirty_epoch() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let workers = (0..2)
            .map(|_| {
                let page = page.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    page.resolve_direct_write_fault().unwrap()
                })
            })
            .collect::<Vec<_>>();

        barrier.wait();
        assert!(workers.into_iter().all(|worker| worker.join().unwrap()));
        assert_eq!(page.cpu_dirty_epoch(), 1);
        assert!(!dependency.remains_current());
    }

    #[test]
    fn dirty_resolution_preserves_each_alias_maximum_permission() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing().clone();
        let host = page.direct_backing().unwrap();
        let arena = DirectArena::new(0x4000).unwrap();
        for (address, maximum) in [
            (0x1000, DirectProtection::ReadWrite),
            (0x2000, DirectProtection::Read),
        ] {
            arena
                .map_pages(&[DirectMapRequest {
                    guest_address: address,
                    backing: &host,
                    protection: DirectProtection::Read,
                }])
                .unwrap();
            page.register_direct_alias(&arena, address, maximum)
                .unwrap();
        }
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();

        assert!(page.resolve_direct_write_fault().unwrap());
        assert!(!dependency.remains_current());
        assert_eq!(
            arena.protection_at(0x1000),
            Some(DirectProtection::ReadWrite)
        );
        assert_eq!(arena.protection_at(0x2000), Some(DirectProtection::Read));
    }

    #[test]
    fn host_batches_invalidate_observers_without_reopening_native_aliases() {
        let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing();
        let host = page.direct_backing().unwrap();
        let arena = DirectArena::new(0x4000).unwrap();
        for address in [0x1000, 0x2000] {
            arena
                .map_pages(&[DirectMapRequest {
                    guest_address: address,
                    backing: &host,
                    protection: DirectProtection::ReadWrite,
                }])
                .unwrap();
            page.register_direct_alias(&arena, address, DirectProtection::ReadWrite)
                .unwrap();
        }
        let mut last = CanonicalCpuWriteDependency::capture(&range).unwrap();
        for value in 1..=32_u8 {
            let mut batch = CanonicalWriteBatch::new();
            batch.stage(&range, 0, &[value]).unwrap();
            batch.commit().unwrap();
            assert!(!last.remains_current());
            assert_eq!(page.cpu_dirty_epoch(), u64::from(value));
            assert!(page.lock_state().cpu_dirty_observer_armed);
            for address in [0x1000, 0x2000] {
                assert_eq!(arena.protection_at(address), Some(DirectProtection::Read));
            }
            last = CanonicalCpuWriteDependency::capture(&range).unwrap();
        }
        // A real native store still opens writable aliases and invalidates the
        // last host snapshot before retrying, rather than faulting forever.
        page.resolve_direct_write_fault().unwrap();
        assert!(!last.remains_current());
        assert_eq!(page.cpu_dirty_epoch(), 33);
        for address in [0x1000, 0x2000] {
            assert_eq!(
                arena.protection_at(address),
                Some(DirectProtection::ReadWrite)
            );
        }
    }

    #[test]
    fn single_page_staging_only_rendezvouses_when_write_protection_needs_arming() {
        for protected in [false, true] {
            let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
            let range = allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap();
            let observer = protected.then(|| CanonicalCpuWriteDependency::capture(&range).unwrap());
            let lease = allocation.inner.store.execution_gate().acquire_shared();
            let (send, receive) = std::sync::mpsc::channel();
            let input = range.clone();
            let worker = thread::spawn(move || {
                let mut batch = CanonicalWriteBatch::new();
                batch.stage(&input, 0, &[0x42]).unwrap();
                send.send(batch).unwrap();
            });
            let early = receive.recv_timeout(std::time::Duration::from_millis(200));
            assert_eq!(early.is_ok(), protected);
            let mut bytes = [0];
            range.segments()[0]
                .backing()
                .load_bytes_quiescent(0, &mut bytes);
            assert_eq!(bytes, [0]);
            drop(lease);
            let batch = early.unwrap_or_else(|_| {
                receive
                    .recv_timeout(std::time::Duration::from_secs(3))
                    .unwrap()
            });
            batch.commit().unwrap();
            worker.join().unwrap();
            allocation.read(0, &mut bytes).unwrap();
            assert_eq!(bytes, [0x42]);
            if let Some(observer) = observer {
                assert!(!observer.remains_current());
            }
        }
    }

    #[test]
    fn canonical_write_batch_commits_across_distinct_stores() {
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
                MappingGeneration::INITIAL,
            )
            .unwrap(),
            CanonicalBackingSegment::new(
                second,
                0,
                4,
                MemoryPermissions::READ_WRITE,
                MappingGeneration::INITIAL,
            )
            .unwrap(),
        ])
        .unwrap();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 2, &[1, 2, 3, 4]).unwrap();

        batch.commit().unwrap();
        let mut observed = [0; 8];
        range.read(0, &mut observed).unwrap();
        assert_eq!(observed, [0, 0, 1, 2, 3, 4, 0, 0]);
    }

    #[test]
    fn canonical_batch_publishes_each_executable_page_to_its_log() {
        for shared_log in [true, false] {
            let first_log = Arc::new(MemoryInvalidationLog::default());
            let second_log = if shared_log {
                first_log.clone()
            } else {
                Arc::new(MemoryInvalidationLog::default())
            };
            let logs = vec![first_log, second_log];
            let mut segments = Vec::new();
            for (index, log) in logs.iter().enumerate() {
                let store = CanonicalBackingStore::allocate().unwrap();
                let page = CanonicalBackingPage::zeroed(
                    &store,
                    GuestPhysicalPageId::new(index as u64 + 1),
                    4,
                    ContentGeneration::INITIAL,
                )
                .unwrap();
                assert!(page.observe_executable_content(log.clone()));
                segments.push(
                    CanonicalBackingSegment::new(
                        page,
                        0,
                        4,
                        MemoryPermissions::READ_WRITE,
                        MappingGeneration::INITIAL,
                    )
                    .unwrap(),
                );
            }
            let range = CanonicalBackingRange::new(segments).unwrap();
            let mut batch = CanonicalWriteBatch::new();
            batch.stage(&range, 0, &[7; 8]).unwrap();
            batch.commit().unwrap();
            for log in &logs {
                let mut records = Vec::new();
                log.read_since(crate::MemoryInvalidationCursor::new(0), &mut records)
                    .unwrap();
                assert_eq!(records.len(), if shared_log { 2 } else { 1 });
                assert!(
                    records
                        .iter()
                        .all(|record| record.origin == MemoryInvalidationOrigin::HostWrite)
                );
            }
            let mut bytes = [0; 8];
            range.read(0, &mut bytes).unwrap();
            assert_eq!(bytes, [7; 8]);
        }
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

        assert_eq!(
            batch.commit(),
            Err(CanonicalWriteBatchError::ConcurrentMutation)
        );
        let mut first = [0xff; 1];
        let mut second = [0xff; 1];
        allocation.read(0x0fff, &mut first).unwrap();
        allocation.read(0x1000, &mut second).unwrap();
        assert_eq!(first, [0]);
        assert_eq!(second, [0x55]);
    }

    #[test]
    fn allocation_write_drains_readers_and_publishes_executable_invalidations() {
        let allocation = CanonicalAllocation::zeroed(12, 4).unwrap();
        let log = Arc::new(MemoryInvalidationLog::default());
        for page in &allocation.inner.pages {
            assert!(page.observe_executable_content(log.clone()));
        }
        allocation.write(0, &[]).unwrap();
        assert_eq!(
            allocation.write(12, &[1]),
            Err(CanonicalAllocationError::InvalidRange)
        );
        let lease = allocation.inner.store.execution_gate().acquire_shared();
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| allocation.write(3, &[7, 8]));
            while !allocation.inner.store.execution_gate().transition_pending() {
                std::thread::yield_now();
            }
            assert_eq!(log.cursor().get(), 0);
            assert_eq!(
                allocation.inner.pages[0].content_generation(),
                ContentGeneration::INITIAL
            );
            drop(lease);
            writer.join().unwrap().unwrap();
        });
        assert!(!allocation.inner.store.execution_gate().transition_pending());
        let mut records = Vec::new();
        log.read_since(crate::MemoryInvalidationCursor::INITIAL, &mut records)
            .unwrap();
        assert_eq!(records.len(), 2);
        for (record, page) in records.iter().zip(&allocation.inner.pages[..2]) {
            assert_eq!(record.origin, MemoryInvalidationOrigin::HostWrite);
            assert_eq!(
                record.kind,
                MemoryInvalidationKind::ExecutableContent {
                    first: page.identity().page(),
                    second: None,
                }
            );
        }
        for page in &allocation.inner.pages[..2] {
            assert_eq!(
                page.content_generation(),
                ContentGeneration::INITIAL.next().unwrap()
            );
        }
        assert_eq!(
            allocation.inner.pages[2].content_generation(),
            ContentGeneration::INITIAL
        );
        let mut bytes = [0; 12];
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn allocation_write_finishes_fallible_page_work_before_copying_any_bytes() {
        let allocation = CanonicalAllocation::zeroed(8, 4).unwrap();
        let second = &allocation.inner.pages[1];
        second
            .inner
            .cpu_dirty_epoch
            .store(u64::MAX, Ordering::Release);
        second.lock_state().cpu_dirty_observer_armed = true;
        assert_eq!(
            allocation.write(3, &[1, 2]),
            Err(CanonicalAllocationError::Page(
                CanonicalPageError::CpuDirtyEpochExhausted
            ))
        );
        let mut first = [0xff; 4];
        allocation.read(0, &mut first).unwrap();
        assert_eq!(first, [0; 4]);
        for page in &allocation.inner.pages {
            assert_eq!(page.content_generation(), ContentGeneration::INITIAL);
        }
        assert!(!allocation.inner.store.execution_gate().transition_pending());
    }

    #[test]
    fn allocation_reads_and_writes_remain_atomic_without_a_transaction_mutex() {
        let allocation = CanonicalAllocation::zeroed(8, 4).unwrap();
        std::thread::scope(|scope| {
            for value in [1, 2] {
                let allocation = &allocation;
                scope.spawn(move || {
                    for _ in 0..200 {
                        allocation.write(0, &[value; 8]).unwrap();
                    }
                });
            }
            for _ in 0..200 {
                let mut bytes = [0; 8];
                allocation.read(0, &mut bytes).unwrap();
                assert!(bytes.iter().all(|byte| *byte == bytes[0]));
            }
        });
    }

    #[test]
    fn canonical_write_batch_rejects_a_native_cpu_write_after_its_snapshot() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        allocation.write(0, &[1]).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].backing();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 0, &[2]).unwrap();

        assert!(page.resolve_direct_write_fault().unwrap());
        let lease = page.store().execution_gate().acquire_shared();
        page.copy_bytes(0, &[3]); // Simulate the repaired native store.
        drop(lease);

        assert_eq!(
            batch.commit(),
            Err(CanonicalWriteBatchError::ConcurrentMutation)
        );
        let mut observed = [0];
        allocation.read(0, &mut observed).unwrap();
        assert_eq!(observed, [3]);
    }

    #[test]
    fn canonical_write_batch_reads_earlier_staged_bytes_without_publishing_them() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 0x0ffe, &[1, 2, 3, 4]).unwrap();

        assert!(!batch.overlaps(&range, 0x100, 0x100).unwrap());
        assert!(batch.overlaps(&range, 0x0fff, 2).unwrap());
        assert!(batch.overlaps(&range, 0x1001, 1).unwrap());

        let mut staged = [0xff; 6];
        batch.read_staged(&range, 0x0ffd, &mut staged).unwrap();
        assert_eq!(staged, [0, 1, 2, 3, 4, 0]);
        let mut canonical = [0xff; 4];
        allocation.read(0x0ffe, &mut canonical).unwrap();
        assert_eq!(canonical, [0; 4]);
    }

    #[test]
    fn staged_reads_do_not_wait_for_native_execution_or_observe_later_cpu_writes() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 0x0ffe, &[1, 2, 3, 4]).unwrap();
        allocation.write(0x0ffd, &[9; 6]).unwrap();
        let gate = allocation.inner.store.execution_gate();
        let lease = gate.acquire_shared();
        let epoch = gate.epoch();
        let (send, receive) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut output = [0xff; 6];
            let result = batch.read_staged(&range, 0x0ffd, &mut output);
            send.send((result, output)).unwrap();
        });
        let result = receive.recv_timeout(std::time::Duration::from_secs(1));
        drop(lease);
        reader.join().unwrap();
        assert_eq!(result.unwrap(), (Ok(()), [0, 1, 2, 3, 4, 0]));
        assert_eq!(gate.epoch(), epoch);
    }
}
