//! Retained canonical RAM backing.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::direct::DirectArenaWeak;
use crate::host_mapped::{HostMappedBacking, HostMappedStore};
use crate::{
    BackingIdentityExhausted, BackingStoreId, CanonicalBackingRange, CanonicalBackingSegment,
    CanonicalPageId, ContentGeneration, CpuVisibilityRequest, DIRECT_PAGE_SIZE,
    DeviceAccessDeclaration, DeviceVisibilityRequest, DirectArena, DirectMemoryError,
    DirectProtectRequest, DirectProtection, DirectStoreControl, ExecutionGate, GenerationExhausted,
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
/// Per-page content generations remain authoritative for aliases and retained
/// ranges.
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
        let host = HostMappedStore::new().map_err(|_| CanonicalPageError::ResourceExhausted)?;
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
    visibility: PageVisibility,
    visibility_epoch: u64,
    cpu_dirty_observer_armed: bool,
    direct_write_epoch: Option<u64>,
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
    write_sequence: AtomicU64,
    direct_write_armed: AtomicU64,
    direct_store_control: OnceLock<DirectStoreControl>,
    executable_invalidations: OnceLock<Arc<MemoryInvalidationLog>>,
    state: Mutex<CanonicalPageState>,
}

struct CanonicalWriter<'a> {
    page: &'a CanonicalBackingPage,
    sequence: u64,
}

impl Drop for CanonicalWriter<'_> {
    fn drop(&mut self) {
        self.page
            .inner
            .write_sequence
            .store(self.sequence.wrapping_add(2), Ordering::Release);
    }
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
                backing: OnceLock::new(),
                generation: AtomicU64::new(generation.get()),
                cpu_dirty_epoch: AtomicU64::new(0),
                write_sequence: AtomicU64::new(0),
                direct_write_armed: AtomicU64::new(0),
                direct_store_control: OnceLock::new(),
                executable_invalidations: OnceLock::new(),
                state: Mutex::new(CanonicalPageState {
                    visibility: PageVisibility::Clean,
                    visibility_epoch: 0,
                    cpu_dirty_observer_armed: false,
                    direct_write_epoch: None,
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
                write_sequence: AtomicU64::new(0),
                direct_write_armed: AtomicU64::new(0),
                direct_store_control: OnceLock::new(),
                executable_invalidations: OnceLock::new(),
                state: Mutex::new(CanonicalPageState {
                    visibility: PageVisibility::Clean,
                    visibility_epoch: 0,
                    cpu_dirty_observer_armed: false,
                    direct_write_epoch: None,
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

    /// Permanently connects device-originated writes on this physical page to
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

    /// Establishes a read-only CPU-write baseline while the backing store's
    /// execution gate is held exclusively by the caller.
    pub(crate) fn arm_cpu_dirty_observer_quiescent(&self) -> Result<u64, CanonicalPageError> {
        self.ensure_backing()?;
        let mut state = self.lock_state();
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
            self.disarm_direct_writes();
            state.cpu_dirty_observer_armed = true;
            if let Err(error) = self.publish_direct_alias_protection(&mut state) {
                state.cpu_dirty_observer_armed = false;
                return Err(CanonicalPageError::Visibility(VisibilityError::HostMemory(
                    error.to_string().into_boxed_str(),
                )));
            }
        }
        Ok(self.cpu_dirty_epoch())
    }

    /// Reports whether native-store publication is armed for this page's
    /// current CPU-visible epoch.
    #[must_use]
    pub fn direct_write_is_armed(&self) -> bool {
        self.inner.direct_write_armed.load(Ordering::Acquire) != 0
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
        state.direct_aliases.insert(
            (arena.identity(), guest_address),
            CanonicalDirectAlias {
                arena: arena.downgrade(),
                guest_address,
                maximum_protection,
            },
        );
        arena.publish_store_control(
            guest_address,
            (maximum_protection == DirectProtection::ReadWrite)
                .then(|| self.direct_store_control()),
        )?;
        self.publish_direct_alias_protection(&mut state)
    }

    /// Removes one derived alias after its host mapping has been revoked.
    pub fn unregister_direct_alias(&self, arena: &DirectArena, guest_address: u64) {
        let _ = arena.publish_store_control(guest_address, None);
        self.lock_state()
            .direct_aliases
            .remove(&(arena.identity(), guest_address));
    }

    /// Arms compact native-store publication for the current CPU visibility
    /// epoch and exposes every semantically writable alias as read/write.
    ///
    /// The page dirty transition has already been published before this is
    /// called. Native stores retain only the temporary pre-Phase-5 content
    /// publication required by `DirectStoreControl`.
    pub fn arm_direct_writes(&self) -> Result<bool, CanonicalPageError> {
        self.ensure_backing()?;
        let mut state = self.lock_state();
        self.arm_direct_writes_locked(&mut state)
    }

    /// Resolves the cold first-write transition before retrying the native
    /// store which faulted. The page becomes CPU-visible and dirty before any
    /// writable alias is republished.
    pub fn resolve_direct_write_fault(&self) -> Result<bool, CanonicalPageError> {
        self.ensure_backing()?;
        loop {
            self.ensure_cpu_visible()
                .map_err(CanonicalPageError::Visibility)?;
            let mut state = self.lock_state();
            match state.visibility {
                PageVisibility::Clean => {
                    self.publish_visibility(&mut state, PageVisibility::CpuNewer)
                        .map_err(CanonicalPageError::Visibility)?;
                }
                PageVisibility::CpuNewer => {}
                PageVisibility::GpuNewer { .. } => continue,
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
            self.publish_cpu_dirty(&mut state)?;
            return self.arm_direct_writes_locked(&mut state);
        }
    }

    fn arm_direct_writes_locked(
        &self,
        state: &mut CanonicalPageState,
    ) -> Result<bool, CanonicalPageError> {
        if state.cpu_dirty_observer_armed || !matches!(state.visibility, PageVisibility::CpuNewer) {
            return Ok(false);
        }
        if state.direct_write_epoch == Some(state.visibility_epoch) {
            return Ok(true);
        }
        state.direct_write_epoch = Some(state.visibility_epoch);
        self.inner.direct_write_armed.store(1, Ordering::Release);
        if let Err(error) = self.publish_direct_alias_protection(state) {
            self.inner.direct_write_armed.store(0, Ordering::Release);
            state.direct_write_epoch = None;
            return Err(CanonicalPageError::Visibility(VisibilityError::HostMemory(
                error.to_string().into_boxed_str(),
            )));
        }
        Ok(true)
    }

    fn direct_store_control(&self) -> &DirectStoreControl {
        self.inner
            .direct_store_control
            .get_or_init(|| DirectStoreControl {
                write_sequence_address: std::ptr::from_ref(&self.inner.write_sequence).addr(),
                generation_address: std::ptr::from_ref(&self.inner.generation).addr(),
                write_armed_address: std::ptr::from_ref(&self.inner.direct_write_armed).addr(),
            })
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

    /// Copies bytes while the caller holds this store's execution gate
    /// exclusively. No ordinary writer can overlap, so no per-store sequence
    /// observation is needed.
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
        match self.lock_state().visibility {
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

    /// Copies bytes and observes their content generation under one stable
    /// page snapshot for checked scalar, code-fetch, and atomic adapters.
    pub fn read_with_generation(
        &self,
        offset: usize,
        output: &mut [u8],
    ) -> Result<ContentGeneration, CanonicalPageError> {
        self.checked_end(offset, output.len())?;
        loop {
            let state = self.lock_state();
            match state.visibility {
                PageVisibility::Clean | PageVisibility::CpuNewer => {
                    return Ok(self.load_bytes_with_generation(offset, output));
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

    /// Captures a checked-write snapshot while this store's execution gate is
    /// held exclusively by the caller. Quiescence lets the page revoke native
    /// write publication directly to its steady readable protection instead
    /// of transiently publishing `PROT_NONE` and restoring it per page.
    fn snapshot_cpu_write_quiescent(
        &self,
    ) -> Result<(Box<[u8]>, ContentGeneration, u64), CanonicalPageError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.size())
            .map_err(|_| CanonicalPageError::ResourceExhausted)?;
        bytes.resize(self.size(), 0);
        let mut state = self.lock_state();
        match state.visibility {
            PageVisibility::Clean | PageVisibility::CpuNewer => {}
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
        self.disarm_direct_writes();
        let Some(next_epoch) = state.visibility_epoch.checked_add(1) else {
            state.visibility = PageVisibility::Invalid;
            return Err(CanonicalPageError::Visibility(
                VisibilityError::VisibilityEpochExhausted,
            ));
        };
        state.visibility_epoch = next_epoch;
        state.direct_write_epoch = None;
        if let Err(error) = self.publish_direct_alias_protection(&mut state) {
            state.visibility = PageVisibility::Invalid;
            return Err(CanonicalPageError::Visibility(VisibilityError::HostMemory(
                error.to_string().into_boxed_str(),
            )));
        }
        self.load_bytes_quiescent(0, &mut bytes);
        let generation = self.content_generation();
        let visibility_epoch = state.visibility_epoch;
        Ok((bytes.into_boxed_slice(), generation, visibility_epoch))
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
        let writer = self.lock_writer();
        let observed = self.content_generation();
        if observed != expected {
            return Err(CanonicalPageError::StaleGeneration { expected, observed });
        }
        self.store_bytes_locked(offset, bytes);
        self.inner.generation.store(next.get(), Ordering::Release);
        drop(writer);
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
        let writer = self.lock_writer();
        let observed = self.content_generation();
        if observed != generation {
            return Err(CanonicalPageError::StaleGeneration {
                expected: generation,
                observed,
            });
        }
        self.store_bytes_locked(offset, bytes);
        drop(writer);
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
                    self.publish_visibility(&mut state, PageVisibility::Conflicting)?;
                    return Err(VisibilityError::ConflictingAccess);
                }
                PageVisibility::Invalid => return Err(VisibilityError::InvalidState),
                PageVisibility::Clean | PageVisibility::CpuNewer => {}
            }
            self.revoke_direct_access(&mut state)?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(self.inner.size)
                .map_err(|_| VisibilityError::ResourceExhausted)?;
            bytes.resize(self.inner.size, 0);
            self.load_bytes_quiescent(0, &mut bytes);
            (bytes, self.content_generation(), state.visibility_epoch)
        };
        let request = DeviceVisibilityRequest {
            page: self.identity(),
            size: self.size(),
            device: declaration.device(),
            visible_at: target,
        };
        if let Err(error) = coordinator.make_device_visible(request, &bytes) {
            let mut state = self.lock_state();
            self.publish_visibility(&mut state, PageVisibility::Invalid)?;
            return Err(VisibilityError::Coordinator(error));
        }
        let mut state = self.lock_state();
        if state.visibility_epoch != epoch || self.content_generation() != generation {
            self.publish_visibility(&mut state, PageVisibility::Conflicting)?;
            return Err(VisibilityError::ConcurrentTransition);
        }
        self.publish_visibility(&mut state, PageVisibility::Clean)
    }

    pub(crate) fn prepare_resident_device_read(
        &self,
        declaration: DeviceAccessDeclaration,
    ) -> Result<(), VisibilityError> {
        let mut state = self.lock_state();
        match state.visibility {
            PageVisibility::Clean => Ok(()),
            PageVisibility::CpuNewer => {
                self.revoke_direct_access(&mut state)?;
                self.publish_visibility(&mut state, PageVisibility::Clean)
            }
            PageVisibility::GpuNewer {
                device, visible_at, ..
            } if device == declaration.device()
                && visible_at <= declaration.device_visible_at() =>
            {
                Ok(())
            }
            PageVisibility::GpuNewer { .. } | PageVisibility::Conflicting => {
                self.publish_visibility(&mut state, PageVisibility::Conflicting)?;
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
        let invalidation = self
            .inner
            .executable_invalidations
            .get()
            .map(|invalidations| {
                invalidations.reserve_with_origin(
                    MemoryInvalidationKind::ExecutableContent {
                        first: self.identity().page(),
                        second: None,
                    },
                    MemoryInvalidationOrigin::DeviceWrite,
                )
            })
            .transpose()
            .map_err(|_| VisibilityError::ResourceExhausted)?;
        let mut state = self.lock_state();
        self.revoke_direct_access(&mut state)?;
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
                self.publish_visibility(&mut state, PageVisibility::Conflicting)?;
                return Err(VisibilityError::ConflictingAccess);
            }
            PageVisibility::Invalid => return Err(VisibilityError::InvalidState),
        }
        self.publish_visibility(
            &mut state,
            PageVisibility::GpuNewer {
                device: declaration.device(),
                visible_at,
                coordinator,
            },
        )?;
        if let Some(invalidation) = invalidation {
            invalidation.commit();
        }
        Ok(())
    }

    pub(crate) fn invalidate_visibility(&self) -> Result<(), VisibilityError> {
        let mut state = self.lock_state();
        self.revoke_direct_access(&mut state)?;
        self.publish_visibility(&mut state, PageVisibility::Invalid)
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
                    let next = match self.content_generation().next() {
                        Ok(next) => next,
                        Err(error) => {
                            self.publish_visibility(&mut state, PageVisibility::Invalid)?;
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
                self.publish_visibility(&mut state, PageVisibility::Invalid)?;
                return Err(VisibilityError::Coordinator(error));
            }
        };
        if bytes.len() != self.size() {
            let observed = bytes.len();
            let mut state = self.lock_state();
            self.publish_visibility(&mut state, PageVisibility::Invalid)?;
            return Err(VisibilityError::IncorrectWritebackSize {
                expected: self.size(),
                observed,
            });
        }
        self.ensure_backing()
            .map_err(|_| VisibilityError::ResourceExhausted)?;
        let mut state = self.lock_state();
        if state.visibility_epoch != epoch {
            self.publish_visibility(&mut state, PageVisibility::Conflicting)?;
            return Err(VisibilityError::ConcurrentTransition);
        }
        self.publish_visibility(&mut state, PageVisibility::Clean)?;
        self.publish_cpu_dirty(&mut state)
            .map_err(|error| match error {
                CanonicalPageError::CpuDirtyEpochExhausted => {
                    VisibilityError::CpuDirtyEpochExhausted
                }
                CanonicalPageError::Visibility(error) => error,
                _ => unreachable!("CPU dirty publication has no other failure mode"),
            })?;
        self.store_bytes(0, &bytes);
        self.inner
            .generation
            .store(next_generation.get(), Ordering::Release);
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
        self.disarm_direct_writes();
        self.publish_direct_alias_protection_for(
            state,
            matches!(visibility, PageVisibility::Clean | PageVisibility::CpuNewer),
        )
        .map_err(|error| {
            state.visibility = PageVisibility::Invalid;
            VisibilityError::HostMemory(error.to_string().into_boxed_str())
        })?;
        let Some(next_epoch) = state.visibility_epoch.checked_add(1) else {
            state.visibility = PageVisibility::Invalid;
            return Err(VisibilityError::VisibilityEpochExhausted);
        };
        state.visibility = visibility;
        state.visibility_epoch = next_epoch;
        state.direct_write_epoch = None;
        Ok(())
    }

    fn publish_cpu_dirty(&self, state: &mut CanonicalPageState) -> Result<(), CanonicalPageError> {
        if !state.cpu_dirty_observer_armed {
            return Ok(());
        }
        let Some(next) = self.cpu_dirty_epoch().checked_add(1) else {
            self.revoke_direct_access(state)
                .map_err(CanonicalPageError::Visibility)?;
            state.visibility = PageVisibility::Invalid;
            return Err(CanonicalPageError::CpuDirtyEpochExhausted);
        };
        state.cpu_dirty_observer_armed = false;
        self.inner.cpu_dirty_epoch.store(next, Ordering::Release);
        Ok(())
    }

    fn revoke_direct_access(&self, state: &mut CanonicalPageState) -> Result<(), VisibilityError> {
        self.disarm_direct_writes();
        self.publish_direct_aliases_as(state, DirectProtection::None)
            .map_err(|error| {
                state.visibility = PageVisibility::Invalid;
                VisibilityError::HostMemory(error.to_string().into_boxed_str())
            })?;
        let Some(next_epoch) = state.visibility_epoch.checked_add(1) else {
            state.visibility = PageVisibility::Invalid;
            return Err(VisibilityError::VisibilityEpochExhausted);
        };
        state.visibility_epoch = next_epoch;
        state.direct_write_epoch = None;
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
        )
    }

    fn publish_direct_alias_protection_for(
        &self,
        state: &mut CanonicalPageState,
        cpu_visible: bool,
    ) -> Result<(), DirectMemoryError> {
        let write_enabled = cpu_visible
            && !state.cpu_dirty_observer_armed
            && self.inner.direct_write_armed.load(Ordering::Acquire) != 0;
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
        for (index, byte) in output.iter_mut().enumerate() {
            let position = offset + index;
            let word_address = backing.base() + (position & !7);
            let word =
                unsafe { AtomicU64::from_ptr(word_address as *mut u64) }.load(Ordering::Acquire);
            *byte = (word >> ((position % 8) * 8)) as u8;
        }
    }

    fn load_bytes_with_generation(&self, offset: usize, output: &mut [u8]) -> ContentGeneration {
        let Some(backing) = self.inner.backing.get() else {
            output.fill(0);
            return self.content_generation();
        };
        loop {
            let before = self.inner.write_sequence.load(Ordering::Acquire);
            if before & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            for (index, byte) in output.iter_mut().enumerate() {
                let position = offset + index;
                let word_address = backing.base() + (position & !7);
                let word = unsafe { AtomicU64::from_ptr(word_address as *mut u64) }
                    .load(Ordering::Acquire);
                *byte = (word >> ((position % 8) * 8)) as u8;
            }
            let generation = self.content_generation();
            if self.inner.write_sequence.load(Ordering::Acquire) == before {
                return generation;
            }
        }
    }

    fn store_bytes(&self, offset: usize, bytes: &[u8]) {
        let writer = self.lock_writer();
        self.store_bytes_locked(offset, bytes);
        drop(writer);
    }

    fn lock_writer(&self) -> CanonicalWriter<'_> {
        let mut sequence = self.inner.write_sequence.load(Ordering::Acquire);
        loop {
            if sequence & 1 != 0 {
                std::hint::spin_loop();
                sequence = self.inner.write_sequence.load(Ordering::Acquire);
                continue;
            }
            match self.inner.write_sequence.compare_exchange_weak(
                sequence,
                sequence.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => sequence = observed,
            }
        }
        CanonicalWriter {
            page: self,
            sequence,
        }
    }

    fn store_bytes_locked(&self, offset: usize, bytes: &[u8]) {
        let backing = self
            .inner
            .backing
            .get()
            .expect("canonical writes materialize host backing during preflight");
        for (index, &byte) in bytes.iter().enumerate() {
            let position = offset + index;
            let word_address = backing.base() + (position & !7);
            let word = unsafe { AtomicU64::from_ptr(word_address as *mut u64) };
            let shift = (position % 8) * 8;
            let mask = !(0xff_u64 << shift);
            let mut observed = word.load(Ordering::Acquire);
            loop {
                let next = (observed & mask) | (u64::from(byte) << shift);
                match word.compare_exchange_weak(
                    observed,
                    next,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(current) => observed = current,
                }
            }
        }
    }

    fn wait_for_direct_writer(&self) {
        while self.inner.write_sequence.load(Ordering::Acquire) & 1 != 0 {
            std::hint::spin_loop();
        }
    }

    /// Prevents a store which observed the previous armed epoch from racing a
    /// subsequent protection downgrade. Native writers validate the flag a
    /// second time after acquiring this page's sequence.
    fn disarm_direct_writes(&self) {
        self.inner.direct_write_armed.store(0, Ordering::Release);
        self.wait_for_direct_writer();
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
        .map_err(|_| CanonicalPageError::ResourceExhausted)
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

    fn prepare_unstaged_cpu_visible(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        end: u64,
    ) -> Result<(), CanonicalWriteBatchError> {
        let mut logical_start = 0_u64;
        let mut visited = BTreeSet::new();
        for segment in range.segments() {
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
            logical_start = logical_end;
            if logical_start >= end {
                break;
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
        let mut logical_start = 0_u64;
        let mut visited = BTreeSet::new();
        for segment in range.segments() {
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
            logical_start = logical_end;
            if logical_start >= end {
                break;
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

        let mut logical_start = 0_u64;
        for segment in range.segments() {
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
            logical_start = logical_end;
            if logical_start >= end {
                break;
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
            for segment in range.segments() {
                stores
                    .entry(segment.backing().store().identity())
                    .or_insert_with(|| segment.backing().store().clone());
            }
            let _transitions = stores
                .values()
                .map(|store| store.execution_gate().acquire_exclusive())
                .collect::<Vec<_>>();
            if !self.unstaged_cpu_visible_quiescent(range, offset, end)? {
                continue;
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
                            .read_quiescent(page_offset, &mut output[copied..copied_end])
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
            return Ok(());
        }
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

        loop {
            self.prepare_unstaged_cpu_visible(range, offset, end)?;

            // Acquire every involved store in stable identity order. Native
            // CPU writers are quiescent for the whole multi-page snapshot, so
            // pages can revoke write publication without a per-page NONE/read
            // protection round trip. Dropping these uncommitted guards
            // preserves the mapping epoch: staging changes no mapping or
            // guest byte.
            let mut stores = BTreeMap::new();
            for segment in range.segments() {
                stores
                    .entry(segment.page().store())
                    .or_insert_with(|| segment.backing().store().clone());
            }
            let _transitions = stores
                .values()
                .map(|store| store.execution_gate().acquire_exclusive())
                .collect::<Vec<_>>();
            if !self.unstaged_cpu_visible_quiescent(range, offset, end)? {
                continue;
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
                                .snapshot_cpu_write_quiescent()
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
            return Ok(());
        }
    }

    /// Publishes every staged byte mutation and advances each page once.
    pub fn commit(self) -> Result<(), CanonicalWriteBatchError> {
        struct Write {
            expected_generation: ContentGeneration,
            expected_visibility_epoch: u64,
            next_generation: ContentGeneration,
            next_visibility_epoch: u64,
            bytes: Option<Box<[u8]>>,
        }

        let mut stores = BTreeMap::new();
        for pending in self.pages.values() {
            stores
                .entry(pending.backing.store().identity())
                .or_insert_with(|| pending.backing.store().clone());
        }
        let _transitions = stores
            .values()
            .map(|store| store.execution_gate().acquire_exclusive())
            .collect::<Vec<_>>();

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
        for ((backing, state), write) in backings.iter().zip(&states).zip(&writes) {
            if backing.content_generation() != write.expected_generation
                || state.visibility_epoch != write.expected_visibility_epoch
                || !matches!(
                    state.visibility,
                    PageVisibility::Clean | PageVisibility::CpuNewer
                )
            {
                return Err(CanonicalWriteBatchError::ConcurrentMutation);
            }
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
                .publish_cpu_dirty(state)
                .expect("CPU dirty epochs were preflighted while page states are locked");
            backing.store_bytes(0, &bytes);
            backing
                .inner
                .generation
                .store(write.next_generation.get(), Ordering::Release);
            state.visibility = PageVisibility::CpuNewer;
            state.visibility_epoch = write.next_visibility_epoch;
            state.direct_write_epoch = None;
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
        let _transaction = self.lock_transaction();
        if bytes.is_empty() {
            return Ok(());
        }
        let first_page = offset / self.inner.page_size;
        let last_page = (end - 1) / self.inner.page_size;
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
        let mut generations = Vec::new();
        generations
            .try_reserve_exact(last_page - first_page + 1)
            .map_err(|_| CanonicalAllocationError::ResourceExhausted)?;
        for page in &self.inner.pages[first_page..=last_page] {
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
        CanonicalCpuWriteDependency, DeviceVisibilityPoint, DirectArena, DirectMapRequest,
        DirectProtection, GenerationKind, VisibilityCoordinatorError, VisibilityState,
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
        range
            .prepare_device_access(declaration, Arc::clone(&coordinator))
            .unwrap();
        range
            .publish_device_write(declaration, coordinator)
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
        range
            .prepare_device_access(declaration, Arc::clone(&coordinator))
            .unwrap();
        range
            .publish_device_write(declaration, coordinator)
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
    fn cpu_write_waits_for_an_in_flight_device_snapshot() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let coordinator = Arc::new(BlockingUploadCoordinator::new());
        let erased: Arc<dyn VisibilityCoordinator> = coordinator.clone();
        let declaration =
            DeviceAccessDeclaration::read(NonCpuDeviceId::new(8), DeviceVisibilityPoint::new(1));
        let cpu_writes = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let worker_range = range.clone();
        let worker = thread::spawn(move || {
            worker_range.prepare_device_access(declaration, Arc::clone(&erased))
        });

        coordinator.entered.wait();
        let writer_allocation = allocation.clone();
        let (writer_started, started) = std::sync::mpsc::channel();
        let (writer_completed, completed) = std::sync::mpsc::channel();
        let writer = thread::spawn(move || {
            writer_started.send(()).unwrap();
            writer_completed
                .send(writer_allocation.write(0, &[0x5a]))
                .unwrap();
        });
        started.recv().unwrap();
        assert_eq!(
            completed.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        );
        coordinator.release.wait();

        assert_eq!(worker.join().unwrap(), Ok(()));
        assert_eq!(completed.recv().unwrap(), Ok(()));
        writer.join().unwrap();
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::CpuNewer
        );
        assert!(!cpu_writes.remains_current());
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
    fn canonical_write_batch_snapshot_rearms_direct_reads_before_stage_returns() {
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
        assert!(page.arm_direct_writes().unwrap());
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
    fn direct_protection_downgrade_waits_for_an_acquired_writer() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let page = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(1),
            DIRECT_PAGE_SIZE,
            ContentGeneration::INITIAL,
        )
        .unwrap();
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
        page.write_preflighted(
            0,
            &[0x5a],
            ContentGeneration::INITIAL,
            ContentGeneration::new(1),
        )
        .unwrap();
        assert!(page.arm_direct_writes().unwrap());
        assert_eq!(
            arena.protection_at(0x1000),
            Some(DirectProtection::ReadWrite)
        );

        let acquired = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let writer_page = page.clone();
        let writer_acquired = Arc::clone(&acquired);
        let writer_release = Arc::clone(&release);
        let writer = thread::spawn(move || {
            let _writer = writer_page.lock_writer();
            writer_acquired.wait();
            writer_release.wait();
        });
        acquired.wait();

        let disarm_page = page.clone();
        let (completed, completion) = std::sync::mpsc::channel();
        let disarmer = thread::spawn(move || {
            let result = disarm_page.arm_cpu_dirty_observer_quiescent();
            completed.send(result).unwrap();
        });
        while page.direct_write_is_armed() {
            thread::yield_now();
        }
        assert!(matches!(
            completion.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        assert_eq!(
            arena.protection_at(0x1000),
            Some(DirectProtection::ReadWrite),
            "the alias cannot be downgraded while an acquired store may use it"
        );

        release.wait();
        writer.join().unwrap();
        completion.recv().unwrap().unwrap();
        disarmer.join().unwrap();
        assert_eq!(arena.protection_at(0x1000), Some(DirectProtection::Read));
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
}
