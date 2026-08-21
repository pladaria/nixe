//! Production process-memory storage.
//!
//! [`ExecutionMemory`] deliberately does not reuse [`super::SyntheticMemory`].
//! The synthetic backend favors deterministic fault injection and simple
//! observability. This backend instead resolves a virtual page through one
//! sparse radix leaf and then indexes a stable physical-page slot directly.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Condvar, Mutex, MutexGuard, PoisonError},
};

use nixe_memory::{
    CanonicalBackingPage, CanonicalBackingRange, CanonicalBackingSegment, CanonicalBackingStore,
    CanonicalPageError, CanonicalRangeTranslationError, CanonicalRangeTranslationErrorReason,
    CanonicalRangeTranslator, ContentMutationEpoch,
};

use crate::{
    address::{
        AddressSpaceId, ContentGeneration, GuestPhysicalPageId, GuestVirtualAddress,
        MappingGeneration,
    },
    error::{InstructionFetchFault, InstructionFetchFaultReason},
    exclusive::ExclusiveReservation,
};

use super::common::{install_error, page_offset};
use super::{
    CodeDependencies, CodePageDependency, CodePageSpan, CpuMemory, DataAccessFault,
    DataAccessFaultReason, DataAccessKind, DataReadResult, DataWriteResult, FetchedCode,
    InstructionMemory, MemoryAccess, MemoryAttributes, MemoryMappingError,
    MemoryMappingErrorReason, MemoryMappingPurpose, MemoryPermissions, MemoryProtectionError,
    MemoryProtectionErrorReason, MemoryQueryResult, MemoryRegionKind, MemoryValue, ProcessMemory,
    SYNTHETIC_PAGE_SIZE, SyntheticInstallError, SyntheticInstallStage, SyntheticMappingInfo,
    SyntheticMmio, SyntheticRamPage,
};

const PAGE_SHIFT: u32 = SYNTHETIC_PAGE_SIZE.trailing_zeros();
const LEAF_BITS: u32 = 9;
const LEAF_ENTRY_COUNT: usize = 1 << LEAF_BITS;
const LEAF_INDEX_MASK: u64 = (LEAF_ENTRY_COUNT as u64) - 1;
const _: () = assert!(SYNTHETIC_PAGE_SIZE.is_power_of_two());

#[derive(Clone, Copy)]
struct ExecutionMapping {
    physical_page: GuestPhysicalPageId,
    physical_slot: usize,
    mapping_generation: MappingGeneration,
    permissions: MemoryPermissions,
    purpose: MemoryMappingPurpose,
    attributes: MemoryAttributes,
}

type PageTableLeaf = [Option<ExecutionMapping>; LEAF_ENTRY_COUNT];

/// Sparse two-level virtual page table.
///
/// One allocated leaf covers 2 MiB of virtual address space. Consequently the
/// allocation cost is proportional to populated regions, never to the 64-bit
/// guest address space. A lookup performs one ordered lookup for the leaf and
/// one array index for the page.
#[derive(Default)]
struct ExecutionPageTable {
    leaves: BTreeMap<(AddressSpaceId, u64), Box<PageTableLeaf>>,
}

impl ExecutionPageTable {
    fn coordinates(virtual_page: u64) -> (u64, usize) {
        (
            virtual_page >> LEAF_BITS,
            (virtual_page & LEAF_INDEX_MASK) as usize,
        )
    }

    fn get(&self, address_space: AddressSpaceId, virtual_page: u64) -> Option<ExecutionMapping> {
        let (leaf, index) = Self::coordinates(virtual_page);
        self.leaves.get(&(address_space, leaf))?[index]
    }

    fn get_mut(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
    ) -> Option<&mut ExecutionMapping> {
        let (leaf, index) = Self::coordinates(virtual_page);
        self.leaves.get_mut(&(address_space, leaf))?[index].as_mut()
    }

    fn insert(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
        mapping: ExecutionMapping,
    ) -> Option<ExecutionMapping> {
        let (leaf, index) = Self::coordinates(virtual_page);
        let entries = self
            .leaves
            .entry((address_space, leaf))
            .or_insert_with(|| Box::new([None; LEAF_ENTRY_COUNT]));
        entries[index].replace(mapping)
    }

    fn remove(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
    ) -> Option<ExecutionMapping> {
        let (leaf, index) = Self::coordinates(virtual_page);
        let key = (address_space, leaf);
        let entries = self.leaves.get_mut(&key)?;
        let removed = entries[index].take();
        if entries.iter().all(Option::is_none) {
            self.leaves.remove(&key);
        }
        removed
    }

    fn mappings(&self) -> impl Iterator<Item = (AddressSpaceId, u64, ExecutionMapping)> + '_ {
        self.leaves
            .iter()
            .flat_map(|(&(address_space, leaf), entries)| {
                entries
                    .iter()
                    .enumerate()
                    .filter_map(move |(index, mapping)| {
                        mapping.map(|mapping| {
                            (address_space, (leaf << LEAF_BITS) | index as u64, mapping)
                        })
                    })
            })
    }
}

enum ExecutionPhysicalPage {
    Ram(CanonicalBackingPage),
    Mmio(Box<dyn SyntheticMmio>),
}

#[derive(Default)]
struct ExecutionMemoryInner {
    // Every published mapping's slot contains a page and its physical ID maps
    // back to that same slot. Aliases intentionally repeat both values. A free
    // slot is absent from all mappings and from `slots_by_id`.
    mappings: ExecutionPageTable,
    physical_slots: Vec<Option<ExecutionPhysicalPage>>,
    free_physical_slots: Vec<usize>,
    slots_by_id: BTreeMap<GuestPhysicalPageId, usize>,
    next_page_id: u64,
    next_mapping_generation: Option<MappingGeneration>,
}

impl ExecutionMemoryInner {
    fn take_mapping_generation(&mut self) -> Option<MappingGeneration> {
        let generation = self.next_mapping_generation?;
        self.next_mapping_generation = generation.next().ok();
        Some(generation)
    }

    fn page(&self, slot: usize) -> Option<&ExecutionPhysicalPage> {
        self.physical_slots.get(slot)?.as_ref()
    }

    fn page_mut(&mut self, slot: usize) -> Option<&mut ExecutionPhysicalPage> {
        self.physical_slots.get_mut(slot)?.as_mut()
    }

    fn push_page(&mut self, id: GuestPhysicalPageId, page: ExecutionPhysicalPage) -> Option<usize> {
        if self.slots_by_id.contains_key(&id) {
            return None;
        }
        let slot = if let Some(slot) = self.free_physical_slots.pop() {
            let destination = self
                .physical_slots
                .get_mut(slot)
                .expect("free physical slot belongs to the slot array");
            debug_assert!(destination.is_none());
            *destination = Some(page);
            slot
        } else {
            let slot = self.physical_slots.len();
            self.physical_slots.push(Some(page));
            slot
        };
        self.slots_by_id.insert(id, slot);
        Some(slot)
    }

    fn remove_page(&mut self, id: GuestPhysicalPageId, slot: usize) {
        let removed_id = self.slots_by_id.remove(&id);
        let removed_page = self.physical_slots.get_mut(slot).and_then(Option::take);
        debug_assert_eq!(removed_id, Some(slot));
        debug_assert!(removed_page.is_some());
        self.free_physical_slots.push(slot);
    }

    fn mapping_at(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Option<ExecutionMapping> {
        self.mappings
            .get(address_space, address.get() >> PAGE_SHIFT)
    }

    fn mapping_state(
        &self,
        address_space: AddressSpaceId,
        virtual_page: u64,
    ) -> Option<(
        MemoryRegionKind,
        MemoryPermissions,
        MemoryMappingPurpose,
        MemoryAttributes,
    )> {
        let mapping = self.mappings.get(address_space, virtual_page)?;
        let region = match self.page(mapping.physical_slot)? {
            ExecutionPhysicalPage::Ram(_) => MemoryRegionKind::Ram,
            ExecutionPhysicalPage::Mmio(_) => MemoryRegionKind::Device,
        };
        Some((
            region,
            mapping.permissions,
            mapping.purpose,
            mapping.attributes,
        ))
    }
}

/// Process-memory backend used by normal guest execution.
///
/// The backend owns production-specific sparse page tables and physical-page
/// slots. It shares public semantic types with [`super::SyntheticMemory`], but
/// neither its storage nor any instruction/data hot path delegates to it.
///
/// Semantic accesses are serialized by one process-memory transaction lock.
/// This intentionally conservative Phase-D policy makes cross-page validation,
/// canonical-page generation changes, MMIO callbacks, and mapping lookup one
/// indivisible operation while permitting the memory object to be shared by
/// future vCPU workers.
pub struct ExecutionMemory {
    backing_store: Option<CanonicalBackingStore>,
    inner: Mutex<ExecutionMemoryInner>,
    lease_state: Mutex<ExecutionLeaseState>,
    lease_changed: Condvar,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MappingEpoch(u64);

impl MappingEpoch {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug)]
struct ExecutionLeaseState {
    active: usize,
    mutating: bool,
    epoch: MappingEpoch,
}

/// RAII proof that one executor may use mappings from the recorded epoch.
/// Mapping mutation waits for every such lease to be released at a safepoint.
pub struct ExecutionMemoryLease<'a> {
    memory: &'a ExecutionMemory,
    epoch: MappingEpoch,
}

impl ExecutionMemoryLease<'_> {
    #[must_use]
    pub const fn epoch(&self) -> MappingEpoch {
        self.epoch
    }
}

impl Drop for ExecutionMemoryLease<'_> {
    fn drop(&mut self) {
        let mut state = self.memory.lock_lease_state();
        state.active = state
            .active
            .checked_sub(1)
            .expect("an execution lease is released exactly once");
        if state.active == 0 {
            self.memory.lease_changed.notify_all();
        }
    }
}

struct MappingMutationGuard<'a> {
    memory: &'a ExecutionMemory,
    state: MutexGuard<'a, ExecutionLeaseState>,
    committed: bool,
}

impl MappingMutationGuard<'_> {
    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for MappingMutationGuard<'_> {
    fn drop(&mut self) {
        if self.committed {
            self.state.epoch = MappingEpoch(
                self.state
                    .epoch
                    .0
                    .checked_add(1)
                    .expect("mapping epoch exhaustion is unreachable in one host run"),
            );
        }
        self.state.mutating = false;
        self.memory.lease_changed.notify_all();
    }
}

impl Default for ExecutionMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionMemory {
    /// Creates an empty production process address space.
    #[must_use]
    pub fn new() -> Self {
        let inner = ExecutionMemoryInner {
            next_page_id: 1,
            next_mapping_generation: Some(MappingGeneration::new(1)),
            ..ExecutionMemoryInner::default()
        };
        Self {
            backing_store: CanonicalBackingStore::allocate().ok(),
            inner: Mutex::new(inner),
            lease_state: Mutex::new(ExecutionLeaseState {
                active: 0,
                mutating: false,
                epoch: MappingEpoch(1),
            }),
            lease_changed: Condvar::new(),
        }
    }

    fn lock_inner(&self) -> MutexGuard<'_, ExecutionMemoryInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn inner_mut(&mut self) -> &mut ExecutionMemoryInner {
        self.inner.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    fn lock_lease_state(&self) -> MutexGuard<'_, ExecutionLeaseState> {
        self.lease_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Acquires one mapping-stable execution lease for a bounded engine slice.
    pub fn acquire_execution_lease(&self) -> ExecutionMemoryLease<'_> {
        let mut state = self.lock_lease_state();
        while state.mutating {
            state = self
                .lease_changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.active = state
            .active
            .checked_add(1)
            .expect("host execution lease count is bounded by active vCPUs");
        let epoch = state.epoch;
        ExecutionMemoryLease {
            memory: self,
            epoch,
        }
    }

    /// Returns the mapping epoch visible to newly acquired execution leases.
    #[must_use]
    pub fn mapping_epoch(&self) -> MappingEpoch {
        self.lock_lease_state().epoch
    }

    /// Reports whether a mapping mutation has closed admission to new engine
    /// slices and is waiting for, or currently owns, quiescence.
    #[must_use]
    pub fn mapping_mutation_pending(&self) -> bool {
        self.lock_lease_state().mutating
    }

    fn begin_mapping_mutation(&self) -> MappingMutationGuard<'_> {
        let mut state = self.lock_lease_state();
        while state.mutating {
            state = self
                .lease_changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        // Publish the mutation intent before waiting for existing executors.
        // Otherwise new leases could continuously overtake a pending mapping
        // change and prevent the coordinator from ever reaching quiescence.
        state.mutating = true;
        while state.active != 0 {
            state = self
                .lease_changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        MappingMutationGuard {
            memory: self,
            state,
            committed: false,
        }
    }

    /// Atomically installs initialized RAM pages.
    pub fn install_ram_pages_atomic(
        &mut self,
        address_space: AddressSpaceId,
        requests: &[SyntheticRamPage<'_>],
    ) -> Result<(), SyntheticInstallError> {
        let backing_store = self.backing_store.clone();
        let inner = self.inner_mut();
        let mut virtual_pages = Vec::with_capacity(requests.len());
        let mut unique_virtual_pages = BTreeSet::new();
        for request in requests {
            if !request
                .virtual_address
                .is_aligned_to(SYNTHETIC_PAGE_SIZE as u64)
            {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "virtual address is not page aligned",
                ));
            }
            if request.bytes.len() != SYNTHETIC_PAGE_SIZE {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "page contents do not match the synthetic page size",
                ));
            }
            if request
                .virtual_address
                .checked_add((SYNTHETIC_PAGE_SIZE - 1) as u64)
                .is_none()
            {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "virtual page range overflows",
                ));
            }
            let virtual_page = request.virtual_address.get() >> PAGE_SHIFT;
            if !unique_virtual_pages.insert(virtual_page) {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "request contains a duplicate virtual page",
                ));
            }
            if inner.mappings.get(address_space, virtual_page).is_some() {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "virtual page is already mapped",
                ));
            }
            virtual_pages.push(virtual_page);
        }
        if requests.is_empty() {
            return Ok(());
        }

        let backing_store = backing_store.ok_or_else(|| {
            install_error(
                SyntheticInstallStage::Allocation,
                requests.first().map(|request| request.virtual_address),
                "canonical backing-store identities are exhausted",
            )
        })?;
        let mut next_page_id = inner.next_page_id;
        let mut pending = Vec::new();
        pending.try_reserve_exact(requests.len()).map_err(|_| {
            install_error(
                SyntheticInstallStage::Allocation,
                requests.first().map(|request| request.virtual_address),
                "host resources are exhausted",
            )
        })?;
        for (index, request) in requests.iter().enumerate() {
            while inner
                .slots_by_id
                .contains_key(&GuestPhysicalPageId::new(next_page_id))
            {
                next_page_id = next_page_id.checked_add(1).ok_or_else(|| {
                    install_error(
                        SyntheticInstallStage::Allocation,
                        Some(request.virtual_address),
                        "physical-page identities are exhausted",
                    )
                })?;
            }
            let physical_page = GuestPhysicalPageId::new(next_page_id);
            next_page_id = next_page_id.checked_add(1).ok_or_else(|| {
                install_error(
                    SyntheticInstallStage::Allocation,
                    Some(request.virtual_address),
                    "physical-page identities are exhausted",
                )
            })?;
            let backing = CanonicalBackingPage::initialized(
                &backing_store,
                physical_page,
                request.bytes,
                ContentGeneration::new(1),
            )
            .map_err(|error| {
                install_error(
                    SyntheticInstallStage::Allocation,
                    Some(request.virtual_address),
                    error.to_string(),
                )
            })?;
            pending.push((
                virtual_pages[index],
                physical_page,
                request.permissions,
                ExecutionPhysicalPage::Ram(backing),
            ));
        }

        let additional_slots = pending
            .len()
            .saturating_sub(inner.free_physical_slots.len());
        inner
            .physical_slots
            .try_reserve(additional_slots)
            .map_err(|_| {
                install_error(
                    SyntheticInstallStage::Allocation,
                    requests.first().map(|request| request.virtual_address),
                    "host resources are exhausted",
                )
            })?;
        let mapping_generation = if pending.is_empty() {
            MappingGeneration::INITIAL
        } else {
            inner.take_mapping_generation().ok_or_else(|| {
                install_error(
                    SyntheticInstallStage::Allocation,
                    requests.first().map(|request| request.virtual_address),
                    "mapping generations are exhausted",
                )
            })?
        };
        for (virtual_page, physical_page, permissions, page) in pending {
            let slot = inner
                .push_page(physical_page, page)
                .expect("preflight allocated a unique physical identity");
            let previous = inner.mappings.insert(
                address_space,
                virtual_page,
                ExecutionMapping {
                    physical_page,
                    physical_slot: slot,
                    mapping_generation,
                    permissions,
                    purpose: MemoryMappingPurpose::Normal,
                    attributes: MemoryAttributes::NONE,
                },
            );
            debug_assert!(previous.is_none());
        }
        inner.next_page_id = next_page_id;
        Ok(())
    }

    /// Returns the observable mapping state used by runtime diagnostics.
    #[must_use]
    pub fn mapping_info(
        &self,
        address_space: AddressSpaceId,
        virtual_address: GuestVirtualAddress,
    ) -> Option<SyntheticMappingInfo> {
        self.lock_inner()
            .mapping_at(address_space, virtual_address)
            .map(|mapping| SyntheticMappingInfo {
                physical_page: mapping.physical_page,
                mapping_generation: mapping.mapping_generation,
                permissions: mapping.permissions,
                attributes: mapping.attributes,
                purpose: mapping.purpose,
            })
    }

    /// Updates the runtime-owned purpose of a complete mapped range.
    pub fn set_mapping_purpose(
        &mut self,
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        size: u64,
        purpose: MemoryMappingPurpose,
    ) -> bool {
        let page_size = SYNTHETIC_PAGE_SIZE as u64;
        if size == 0 || !start.is_aligned_to(page_size) || !size.is_multiple_of(page_size) {
            return false;
        }
        let Some(end) = start.get().checked_add(size) else {
            return false;
        };
        let first_page = start.get() >> PAGE_SHIFT;
        let end_page = end >> PAGE_SHIFT;
        let inner = self.inner_mut();
        if !(first_page..end_page).all(|page| inner.mappings.get(address_space, page).is_some()) {
            return false;
        }
        let Some(mapping_generation) = inner.take_mapping_generation() else {
            return false;
        };
        for page in first_page..end_page {
            let mapping = inner
                .mappings
                .get_mut(address_space, page)
                .expect("range was preflighted");
            mapping.purpose = purpose;
            mapping.mapping_generation = mapping_generation;
        }
        true
    }

    /// Returns the number of physical pages owned by this backend.
    #[must_use]
    pub fn physical_page_count(&self) -> usize {
        self.lock_inner().slots_by_id.len()
    }

    /// Returns the store-wide epoch of the latest published content mutation.
    ///
    /// This is an O(1) change detector. Per-page content generations remain
    /// authoritative for code dependencies, aliases, and retained ranges.
    #[must_use]
    pub fn content_mutation_epoch(&self) -> ContentMutationEpoch {
        self.backing_store.as_ref().map_or(
            ContentMutationEpoch::INITIAL,
            CanonicalBackingStore::content_epoch,
        )
    }

    /// Creates a zero-filled RAM page for explicit runtime or differential setup.
    pub fn add_ram_page(&mut self, page: GuestPhysicalPageId) -> bool {
        let Some(store) = self.backing_store.clone() else {
            return false;
        };
        let inner = self.inner_mut();
        let Ok(backing) = CanonicalBackingPage::initialized(
            &store,
            page,
            &[0; SYNTHETIC_PAGE_SIZE],
            ContentGeneration::INITIAL,
        ) else {
            return false;
        };
        inner
            .push_page(page, ExecutionPhysicalPage::Ram(backing))
            .is_some()
    }

    /// Creates a device-backed physical page.
    pub fn add_mmio_page(
        &mut self,
        page: GuestPhysicalPageId,
        handler: impl SyntheticMmio + 'static,
    ) -> bool {
        self.inner_mut()
            .push_page(page, ExecutionPhysicalPage::Mmio(Box::new(handler)))
            .is_some()
    }

    /// Copies initialization bytes into RAM and advances its generation.
    pub fn initialize_ram(
        &mut self,
        page: GuestPhysicalPageId,
        offset: usize,
        bytes: &[u8],
    ) -> bool {
        let inner = self.inner_mut();
        let Some(&slot) = inner.slots_by_id.get(&page) else {
            return false;
        };
        let Some(ExecutionPhysicalPage::Ram(backing)) = inner.page(slot) else {
            return false;
        };
        let Some(end) = offset.checked_add(bytes.len()) else {
            return false;
        };
        if end > SYNTHETIC_PAGE_SIZE {
            return false;
        }
        if backing.prepare_write().is_err() {
            return false;
        }
        let generation = backing.content_generation();
        let Ok(next_generation) = generation.next() else {
            return false;
        };
        backing
            .write_preflighted(offset, bytes, generation, next_generation)
            .is_ok()
    }

    /// Overwrites mapped RAM from a trusted host producer, ignoring guest
    /// write permissions while retaining mapping and region validation.
    ///
    /// This is used for kernel-owned shared-memory producers whose guest view
    /// is intentionally read-only.
    pub fn overwrite_mapped_ram(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        bytes: &[u8],
    ) -> bool {
        self.write_mapped_ram_checked(address_space, address, bytes, false)
    }

    /// Atomically writes a fully validated guest-writable RAM range. No byte
    /// is changed when any page is unmapped, non-RAM, read-only, or unable to
    /// publish its next content generation.
    pub fn write_mapped_ram(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        bytes: &[u8],
    ) -> bool {
        self.write_mapped_ram_checked(address_space, address, bytes, true)
    }

    fn write_mapped_ram_checked(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        bytes: &[u8],
        require_write: bool,
    ) -> bool {
        let Some(end) = address.get().checked_add(bytes.len() as u64) else {
            return false;
        };
        let inner = self.lock_inner();
        let mut cursor = address.get();
        let mut pending_generations = BTreeMap::new();
        while cursor < end {
            let virtual_address = GuestVirtualAddress::new(cursor);
            let Some(mapping) = inner.mapping_at(address_space, virtual_address) else {
                return false;
            };
            if require_write && !mapping.permissions.contains(MemoryPermissions::WRITE) {
                return false;
            }
            let Some(ExecutionPhysicalPage::Ram(backing)) = inner.page(mapping.physical_slot)
            else {
                return false;
            };
            if let std::collections::btree_map::Entry::Vacant(entry) =
                pending_generations.entry(mapping.physical_slot)
            {
                if backing.prepare_write().is_err() {
                    return false;
                }
                let current = backing.content_generation();
                let Ok(next) = current.next() else {
                    return false;
                };
                entry.insert((current, next));
            }
            let remaining_in_page = SYNTHETIC_PAGE_SIZE - page_offset(virtual_address);
            cursor = cursor.saturating_add(remaining_in_page.min((end - cursor) as usize) as u64);
        }

        let mut copied = 0;
        let mut written_slots = BTreeSet::new();
        while copied < bytes.len() {
            let virtual_address =
                GuestVirtualAddress::new(address.get().saturating_add(copied as u64));
            let mapping = inner
                .mapping_at(address_space, virtual_address)
                .expect("host overwrite range was validated");
            let offset = page_offset(virtual_address);
            let count = (SYNTHETIC_PAGE_SIZE - offset).min(bytes.len() - copied);
            let Some(ExecutionPhysicalPage::Ram(backing)) = inner.page(mapping.physical_slot)
            else {
                unreachable!("host overwrite RAM range was validated")
            };
            let (expected, next) = pending_generations[&mapping.physical_slot];
            let result = if written_slots.insert(mapping.physical_slot) {
                backing.write_preflighted(offset, &bytes[copied..copied + count], expected, next)
            } else {
                backing.write_fragment_preflighted(offset, &bytes[copied..copied + count], next)
            };
            if result.is_err() {
                return false;
            }
            copied += count;
        }
        true
    }

    /// Publishes an alias mapping for an existing physical page.
    pub fn map_page(
        &mut self,
        address_space: AddressSpaceId,
        virtual_address: GuestVirtualAddress,
        physical_page: GuestPhysicalPageId,
        permissions: MemoryPermissions,
    ) -> bool {
        if !virtual_address.is_aligned_to(SYNTHETIC_PAGE_SIZE as u64) {
            return false;
        }
        let inner = self.inner_mut();
        let Some(&physical_slot) = inner.slots_by_id.get(&physical_page) else {
            return false;
        };
        let virtual_page = virtual_address.get() >> PAGE_SHIFT;
        if inner.mappings.get(address_space, virtual_page).is_some() {
            return false;
        }
        let Some(mapping_generation) = inner.take_mapping_generation() else {
            return false;
        };
        let previous = inner.mappings.insert(
            address_space,
            virtual_page,
            ExecutionMapping {
                physical_page,
                physical_slot,
                mapping_generation,
                permissions,
                purpose: MemoryMappingPurpose::Normal,
                attributes: MemoryAttributes::NONE,
            },
        );
        debug_assert!(previous.is_none());
        true
    }

    fn fetch<const N: usize>(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        alignment: u8,
    ) -> Result<([u8; N], CodeDependencies), InstructionFetchFault> {
        if !address.is_aligned_to(u64::from(alignment)) {
            return Err(InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::Misaligned {
                    required_alignment: alignment,
                },
            ));
        }
        let inner = self.lock_inner();
        let end_offset = page_offset(address) + N;
        // The only callers request aligned 2-byte or 4-byte architectural
        // encodings. Both widths divide the page size, so a valid fetch cannot
        // cross a page. T32 instructions spanning pages are assembled by
        // `fetch_t32_32` from two independently checked halfword fetches.
        debug_assert!(end_offset <= SYNTHETIC_PAGE_SIZE);
        let mapping = inner.mapping_at(address_space, address).ok_or_else(|| {
            InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::Unmapped,
            )
        })?;
        if !mapping.permissions.contains(MemoryPermissions::EXECUTE) {
            return Err(InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::ExecutePermissionDenied,
            ));
        }
        let Some(ExecutionPhysicalPage::Ram(backing)) = inner.page(mapping.physical_slot) else {
            return Err(InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::Memory("executable mapping is not RAM".into()),
            ));
        };
        let mut bytes = [0; N];
        let generation = backing
            .read_with_generation(page_offset(address), &mut bytes)
            .map_err(|reason| {
                InstructionFetchFault::new(
                    address_space,
                    address,
                    InstructionFetchFaultReason::Memory(reason.to_string().into()),
                )
            })?;
        Ok((
            bytes,
            CodeDependencies::one(CodePageDependency {
                page: mapping.physical_page,
                generation,
                mapping_generation: mapping.mapping_generation,
            }),
        ))
    }
}

impl InstructionMemory for ExecutionMemory {
    fn code_page_span(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<CodePageSpan, InstructionFetchFault> {
        let page_start = GuestVirtualAddress::new(address.get() >> PAGE_SHIFT << PAGE_SHIFT);
        let inner = self.lock_inner();
        let mapping = inner.mapping_at(address_space, address).ok_or_else(|| {
            InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::Unmapped,
            )
        })?;
        if !mapping.permissions.contains(MemoryPermissions::EXECUTE) {
            return Err(InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::ExecutePermissionDenied,
            ));
        }
        let end_exclusive = page_start.checked_add(SYNTHETIC_PAGE_SIZE as u64);
        Ok(CodePageSpan::containing(page_start, end_exclusive, address)
            .expect("production page arithmetic contains its source address"))
    }

    fn fetch16(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u16>, InstructionFetchFault> {
        let (bytes, dependencies) = self.fetch::<2>(address_space, address, 2)?;
        Ok(FetchedCode {
            bits: u16::from_le_bytes(bytes),
            dependencies,
        })
    }

    fn fetch32(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault> {
        let (bytes, dependencies) = self.fetch::<4>(address_space, address, 4)?;
        Ok(FetchedCode {
            bits: u32::from_le_bytes(bytes),
            dependencies,
        })
    }
}

impl CanonicalRangeTranslator for ExecutionMemory {
    fn translate_canonical_range(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        size: u64,
        required_permissions: MemoryPermissions,
    ) -> Result<CanonicalBackingRange, CanonicalRangeTranslationError> {
        let failure = |address, reason| CanonicalRangeTranslationError {
            address_space,
            address,
            reason,
        };
        if size == 0 {
            return Err(failure(
                address,
                CanonicalRangeTranslationErrorReason::Empty,
            ));
        }
        if address.checked_add(size - 1).is_none() {
            return Err(failure(
                address,
                CanonicalRangeTranslationErrorReason::AddressOverflow,
            ));
        }

        let page_size = SYNTHETIC_PAGE_SIZE as u64;
        let covered_bytes = (page_offset(address) as u64)
            .checked_add(size)
            .ok_or_else(|| {
                failure(
                    address,
                    CanonicalRangeTranslationErrorReason::AddressOverflow,
                )
            })?;
        let segment_capacity =
            usize::try_from(covered_bytes.div_ceil(page_size)).map_err(|_| {
                failure(
                    address,
                    CanonicalRangeTranslationErrorReason::ResourceExhausted,
                )
            })?;
        let mut segments = Vec::new();
        segments.try_reserve_exact(segment_capacity).map_err(|_| {
            failure(
                address,
                CanonicalRangeTranslationErrorReason::ResourceExhausted,
            )
        })?;

        let inner = self.lock_inner();
        let mut cursor = address;
        let mut remaining = size;
        while remaining != 0 {
            let mapping = inner
                .mapping_at(address_space, cursor)
                .ok_or_else(|| failure(cursor, CanonicalRangeTranslationErrorReason::Unmapped))?;
            if !mapping.permissions.contains(required_permissions) {
                return Err(failure(
                    cursor,
                    CanonicalRangeTranslationErrorReason::PermissionDenied,
                ));
            }
            let backing = match inner.page(mapping.physical_slot) {
                Some(ExecutionPhysicalPage::Ram(backing)) => backing.clone(),
                Some(ExecutionPhysicalPage::Mmio(_)) => {
                    return Err(failure(
                        cursor,
                        CanonicalRangeTranslationErrorReason::DeviceMemory,
                    ));
                }
                None => {
                    return Err(failure(
                        cursor,
                        CanonicalRangeTranslationErrorReason::InconsistentBacking,
                    ));
                }
            };
            let offset = page_offset(cursor) as u64;
            let count = remaining.min(page_size - offset);
            let generation = backing.content_generation();
            let segment = CanonicalBackingSegment::new(
                backing,
                offset,
                count,
                mapping.permissions,
                generation,
                mapping.mapping_generation,
            )
            .map_err(|_| {
                failure(
                    cursor,
                    CanonicalRangeTranslationErrorReason::InconsistentBacking,
                )
            })?;
            segments.push(segment);
            remaining -= count;
            if remaining != 0 {
                cursor = cursor.checked_add(count).ok_or_else(|| {
                    failure(
                        cursor,
                        CanonicalRangeTranslationErrorReason::AddressOverflow,
                    )
                })?;
            }
        }
        CanonicalBackingRange::new(segments).map_err(|_| {
            failure(
                address,
                CanonicalRangeTranslationErrorReason::InconsistentBacking,
            )
        })
    }
}

#[derive(Clone, Copy)]
struct ResolvedExecutionAccess {
    first: ExecutionMapping,
    second: Option<ExecutionMapping>,
    first_bytes: usize,
    region: MemoryRegionKind,
}

fn resolve_data_access(
    inner: &ExecutionMemoryInner,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    access: MemoryAccess,
    kind: DataAccessKind,
) -> Result<ResolvedExecutionAccess, DataAccessFault> {
    let required_alignment = access.alignment.bytes(access.size);
    if !address.is_aligned_to(u64::from(required_alignment)) {
        return Err(DataAccessFault::new(
            address_space,
            address,
            kind,
            DataAccessFaultReason::Misaligned { required_alignment },
        ));
    }
    let byte_count = access.size.bytes();
    if address.checked_add((byte_count - 1) as u64).is_none() {
        return Err(DataAccessFault::new(
            address_space,
            address,
            kind,
            DataAccessFaultReason::AddressOverflow,
        ));
    }
    let first_bytes = (SYNTHETIC_PAGE_SIZE - page_offset(address)).min(byte_count);
    let second_address = (first_bytes < byte_count).then_some(
        address
            .checked_add(first_bytes as u64)
            .expect("validated access end contains its second page"),
    );
    let required = match kind {
        DataAccessKind::Read => MemoryPermissions::READ,
        DataAccessKind::Write => MemoryPermissions::WRITE,
    };
    let resolve = |current| {
        let mapping = inner.mapping_at(address_space, current).ok_or_else(|| {
            DataAccessFault::new(
                address_space,
                current,
                kind,
                DataAccessFaultReason::Unmapped,
            )
        })?;
        if !mapping.permissions.contains(required) {
            let permission_fault = match kind {
                DataAccessKind::Read => DataAccessFaultReason::ReadPermissionDenied,
                DataAccessKind::Write => DataAccessFaultReason::WritePermissionDenied,
            };
            return Err(DataAccessFault::new(
                address_space,
                current,
                kind,
                permission_fault,
            ));
        }
        let region = match inner.page(mapping.physical_slot) {
            Some(ExecutionPhysicalPage::Ram(_)) => MemoryRegionKind::Ram,
            Some(ExecutionPhysicalPage::Mmio(_)) => MemoryRegionKind::Device,
            None => {
                return Err(DataAccessFault::new(
                    address_space,
                    current,
                    kind,
                    DataAccessFaultReason::Unmapped,
                ));
            }
        };
        Ok((mapping, region))
    };
    let (first, region) = resolve(address)?;
    let second = if let Some(second_address) = second_address {
        let (second, second_region) = resolve(second_address)?;
        if region != second_region {
            return Err(DataAccessFault::new(
                address_space,
                second_address,
                kind,
                DataAccessFaultReason::MixedRegions,
            ));
        }
        Some(second)
    } else {
        None
    };
    Ok(ResolvedExecutionAccess {
        first,
        second,
        first_bytes,
        region,
    })
}

impl CpuMemory for ExecutionMemory {
    fn read(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault> {
        let mut inner = self.lock_inner();
        let resolved =
            resolve_data_access(&inner, address_space, address, access, DataAccessKind::Read)?;
        if resolved.region == MemoryRegionKind::Device {
            if resolved.second.is_some() {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Read,
                    DataAccessFaultReason::MixedRegions,
                ));
            }
            let ExecutionPhysicalPage::Mmio(handler) = inner
                .page_mut(resolved.first.physical_slot)
                .expect("resolved device page exists")
            else {
                unreachable!()
            };
            let value = handler
                .read(page_offset(address) as u64, access)
                .map_err(|reason| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Read,
                        DataAccessFaultReason::Device(reason),
                    )
                })?;
            if value.size() != access.size {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Read,
                    DataAccessFaultReason::ValueSizeMismatch,
                ));
            }
            return Ok(DataReadResult {
                value,
                region: MemoryRegionKind::Device,
            });
        }

        let byte_count = access.size.bytes();
        let mut bytes = [0_u8; 16];
        match resolved.second {
            None => {
                let ExecutionPhysicalPage::Ram(backing) = inner
                    .page(resolved.first.physical_slot)
                    .expect("resolved RAM page exists")
                else {
                    unreachable!()
                };
                backing
                    .read(page_offset(address), &mut bytes[..byte_count])
                    .map_err(|reason| {
                        DataAccessFault::new(
                            address_space,
                            address,
                            DataAccessKind::Read,
                            DataAccessFaultReason::HostBacking(reason.to_string().into()),
                        )
                    })?;
            }
            Some(second) => {
                let mappings = [resolved.first, second];
                let mut copied = 0;
                for (mapping, count, offset) in [
                    (mappings[0], resolved.first_bytes, page_offset(address)),
                    (mappings[1], byte_count - resolved.first_bytes, 0),
                ] {
                    let ExecutionPhysicalPage::Ram(backing) = inner
                        .page(mapping.physical_slot)
                        .expect("resolved RAM page exists")
                    else {
                        unreachable!()
                    };
                    backing
                        .read(offset, &mut bytes[copied..copied + count])
                        .map_err(|reason| {
                            DataAccessFault::new(
                                address_space,
                                address,
                                DataAccessKind::Read,
                                DataAccessFaultReason::HostBacking(reason.to_string().into()),
                            )
                        })?;
                    copied += count;
                }
            }
        }
        Ok(DataReadResult {
            value: MemoryValue::from_le_slice(access.size, &bytes[..byte_count]),
            region: MemoryRegionKind::Ram,
        })
    }

    fn write(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<DataWriteResult, DataAccessFault> {
        if value.size() != access.size {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ValueSizeMismatch,
            ));
        }
        let mut inner = self.lock_inner();
        let resolved = resolve_data_access(
            &inner,
            address_space,
            address,
            access,
            DataAccessKind::Write,
        )?;
        if resolved.region == MemoryRegionKind::Device {
            if resolved.second.is_some() {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::MixedRegions,
                ));
            }
            let ExecutionPhysicalPage::Mmio(handler) = inner
                .page_mut(resolved.first.physical_slot)
                .expect("resolved device page exists")
            else {
                unreachable!()
            };
            handler
                .write(page_offset(address) as u64, access, value)
                .map_err(|reason| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::Device(reason),
                    )
                })?;
            return Ok(DataWriteResult {
                region: MemoryRegionKind::Device,
            });
        }

        let byte_count = access.size.bytes();
        let mut bytes = [0_u8; 16];
        value.copy_le_bytes(&mut bytes[..byte_count]);
        match resolved.second {
            None => {
                let ExecutionPhysicalPage::Ram(backing) = inner
                    .page(resolved.first.physical_slot)
                    .expect("resolved RAM page exists")
                else {
                    unreachable!()
                };
                backing.prepare_write().map_err(|reason| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::HostBacking(reason.to_string().into()),
                    )
                })?;
                let generation = backing.content_generation();
                let next_generation = generation.next().map_err(|_| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::ContentGenerationExhausted,
                    )
                })?;
                let offset = page_offset(address);
                backing
                    .write_preflighted(offset, &bytes[..byte_count], generation, next_generation)
                    .map_err(|reason| {
                        DataAccessFault::new(
                            address_space,
                            address,
                            DataAccessKind::Write,
                            DataAccessFaultReason::HostBacking(reason.to_string().into()),
                        )
                    })?;
            }
            Some(second) => {
                let mappings = [resolved.first, second];
                let backing = |slot| match inner.page(slot) {
                    Some(ExecutionPhysicalPage::Ram(backing)) => backing,
                    _ => unreachable!("resolved RAM page exists"),
                };
                let first_backing = backing(mappings[0].physical_slot);
                first_backing.prepare_write().map_err(|reason| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::HostBacking(reason.to_string().into()),
                    )
                })?;
                let first_generation = first_backing.content_generation();
                let next_first = first_generation.next().map_err(|_| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::ContentGenerationExhausted,
                    )
                })?;
                let distinct_pages = mappings[1].physical_slot != mappings[0].physical_slot;
                let (second_generation, next_second) = if distinct_pages {
                    let second_backing = backing(mappings[1].physical_slot);
                    second_backing.prepare_write().map_err(|reason| {
                        DataAccessFault::new(
                            address_space,
                            address,
                            DataAccessKind::Write,
                            DataAccessFaultReason::HostBacking(reason.to_string().into()),
                        )
                    })?;
                    let generation = second_backing.content_generation();
                    let next = generation.next().map_err(|_| {
                        DataAccessFault::new(
                            address_space,
                            address,
                            DataAccessKind::Write,
                            DataAccessFaultReason::ContentGenerationExhausted,
                        )
                    })?;
                    (Some(generation), Some(next))
                } else {
                    (None, None)
                };
                let mut copied = 0;
                for (index, (mapping, count, offset)) in [
                    (mappings[0], resolved.first_bytes, page_offset(address)),
                    (mappings[1], byte_count - resolved.first_bytes, 0),
                ]
                .into_iter()
                .enumerate()
                {
                    let page = backing(mapping.physical_slot);
                    let result = if index == 0 {
                        page.write_preflighted(
                            offset,
                            &bytes[copied..copied + count],
                            first_generation,
                            next_first,
                        )
                    } else if let (Some(generation), Some(next)) = (second_generation, next_second)
                    {
                        page.write_preflighted(
                            offset,
                            &bytes[copied..copied + count],
                            generation,
                            next,
                        )
                    } else {
                        page.write_fragment_preflighted(
                            offset,
                            &bytes[copied..copied + count],
                            next_first,
                        )
                    };
                    result.map_err(|reason| {
                        DataAccessFault::new(
                            address_space,
                            address,
                            DataAccessKind::Write,
                            DataAccessFaultReason::HostBacking(reason.to_string().into()),
                        )
                    })?;
                    copied += count;
                }
            }
        }
        Ok(DataWriteResult {
            region: MemoryRegionKind::Ram,
        })
    }

    fn query_memory(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        end_exclusive: GuestVirtualAddress,
    ) -> Option<MemoryQueryResult> {
        if address.get() >= end_exclusive.get() {
            return None;
        }
        let inner = self.lock_inner();
        let page_size = SYNTHETIC_PAGE_SIZE as u64;
        let page = address.get() >> PAGE_SHIFT;
        let end_page = end_exclusive.get() >> PAGE_SHIFT;
        let state = inner.mapping_state(address_space, page);

        let (first_page, last_page_exclusive) = if let Some(state) = state {
            let mut first = page;
            while first > 0 && inner.mapping_state(address_space, first - 1) == Some(state) {
                first -= 1;
            }
            let mut last = page + 1;
            while last < end_page && inner.mapping_state(address_space, last) == Some(state) {
                last += 1;
            }
            (first, last)
        } else {
            let mut previous = 0;
            let mut next = end_page;
            for (space, mapped_page, _) in inner.mappings.mappings() {
                if space != address_space {
                    continue;
                }
                if mapped_page < page {
                    previous = previous.max(mapped_page.saturating_add(1));
                } else if mapped_page > page {
                    next = next.min(mapped_page);
                }
            }
            (previous.min(page), next.max(page + 1))
        };
        let base = first_page.checked_mul(page_size)?;
        let end = last_page_exclusive.checked_mul(page_size)?;
        let (region, permissions, purpose, attributes) = state
            .map(|(region, permissions, purpose, attributes)| {
                (Some(region), permissions, purpose, attributes)
            })
            .unwrap_or((
                None,
                MemoryPermissions::NONE,
                MemoryMappingPurpose::Normal,
                MemoryAttributes::NONE,
            ));
        Some(MemoryQueryResult {
            base: GuestVirtualAddress::new(base),
            size: end.checked_sub(base)?,
            region,
            permissions,
            purpose,
            attributes,
        })
    }

    fn load_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<(DataReadResult, ExclusiveReservation), DataAccessFault> {
        let inner = self.lock_inner();
        let resolved =
            resolve_data_access(&inner, address_space, address, access, DataAccessKind::Read)?;
        if resolved.second.is_some() {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::MixedRegions,
            ));
        }
        let mapping = resolved.first;
        let ExecutionPhysicalPage::Ram(backing) = inner
            .page(mapping.physical_slot)
            .expect("mapping references a page")
        else {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::MixedRegions,
            ));
        };
        let byte_count = access.size.bytes();
        let mut bytes = [0_u8; 16];
        let generation = backing
            .read_with_generation(page_offset(address), &mut bytes[..byte_count])
            .map_err(|reason| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Read,
                    DataAccessFaultReason::HostBacking(reason.to_string().into()),
                )
            })?;
        Ok((
            DataReadResult {
                value: MemoryValue::from_le_slice(access.size, &bytes[..byte_count]),
                region: MemoryRegionKind::Ram,
            },
            ExclusiveReservation {
                page: mapping.physical_page,
                byte_offset: page_offset(address) as u16,
                access_size: access.size.bytes() as u8,
                generation,
            },
        ))
    }

    fn store_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
        reservation: ExclusiveReservation,
    ) -> Result<(DataWriteResult, bool), DataAccessFault> {
        if value.size() != access.size {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ValueSizeMismatch,
            ));
        }
        let inner = self.lock_inner();
        let resolved = resolve_data_access(
            &inner,
            address_space,
            address,
            access,
            DataAccessKind::Write,
        )?;
        if resolved.second.is_some() {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::MixedRegions,
            ));
        }
        let Some(ExecutionPhysicalPage::Ram(backing)) = inner.page(resolved.first.physical_slot)
        else {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::MixedRegions,
            ));
        };
        backing.prepare_cpu_access().map_err(|reason| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::HostBacking(reason.to_string().into()),
            )
        })?;
        let generation = backing.content_generation();
        let matches = reservation.page == resolved.first.physical_page
            && usize::from(reservation.byte_offset) == page_offset(address)
            && usize::from(reservation.access_size) == access.size.bytes()
            && reservation.generation == generation;
        if !matches {
            return Ok((
                DataWriteResult {
                    region: MemoryRegionKind::Ram,
                },
                false,
            ));
        }
        backing.prepare_write().map_err(|reason| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::HostBacking(reason.to_string().into()),
            )
        })?;
        let next_generation = generation.next().map_err(|_| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ContentGenerationExhausted,
            )
        })?;
        let mut bytes = [0_u8; 16];
        let byte_count = access.size.bytes();
        value.copy_le_bytes(&mut bytes[..byte_count]);
        match backing.write_preflighted(
            page_offset(address),
            &bytes[..byte_count],
            generation,
            next_generation,
        ) {
            Ok(()) => {}
            Err(CanonicalPageError::StaleGeneration { .. }) => {
                return Ok((
                    DataWriteResult {
                        region: MemoryRegionKind::Ram,
                    },
                    false,
                ));
            }
            Err(reason) => {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::HostBacking(reason.to_string().into()),
                ));
            }
        }
        Ok((
            DataWriteResult {
                region: MemoryRegionKind::Ram,
            },
            true,
        ))
    }
}

impl ProcessMemory for ExecutionMemory {
    fn resize_zeroed_mapping(
        &self,
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        old_size: u64,
        new_size: u64,
        permissions: MemoryPermissions,
        purpose: MemoryMappingPurpose,
    ) -> Result<(), MemoryMappingError> {
        let page_size = SYNTHETIC_PAGE_SIZE as u64;
        let error = |address, reason| MemoryMappingError {
            address_space,
            address,
            reason,
        };
        if !start.is_aligned_to(page_size)
            || !old_size.is_multiple_of(page_size)
            || !new_size.is_multiple_of(page_size)
            || start.get().checked_add(old_size.max(new_size)).is_none()
        {
            return Err(error(start, MemoryMappingErrorReason::InvalidRange));
        }
        if permissions.contains(MemoryPermissions::WRITE)
            && permissions.contains(MemoryPermissions::EXECUTE)
        {
            return Err(error(start, MemoryMappingErrorReason::WritableExecutable));
        }

        let first_page = start.get() >> PAGE_SHIFT;
        let old_end_page = first_page + old_size / page_size;
        let new_end_page = first_page + new_size / page_size;
        let backing_store = self.backing_store.clone();
        let mut mutation = self.begin_mapping_mutation();
        let mut inner = self.lock_inner();
        for page in first_page..old_end_page {
            let Some(mapping) = inner.mappings.get(address_space, page) else {
                return Err(error(
                    GuestVirtualAddress::new(page * page_size),
                    MemoryMappingErrorReason::MappingStateMismatch,
                ));
            };
            if mapping.purpose != purpose || mapping.permissions != permissions {
                return Err(error(
                    GuestVirtualAddress::new(page * page_size),
                    MemoryMappingErrorReason::MappingStateMismatch,
                ));
            }
        }
        for page in old_end_page..new_end_page {
            if inner.mappings.get(address_space, page).is_some() {
                return Err(error(
                    GuestVirtualAddress::new(page * page_size),
                    MemoryMappingErrorReason::AlreadyMapped,
                ));
            }
        }

        if new_end_page < old_end_page {
            let removed_pages = usize::try_from(old_end_page - new_end_page)
                .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
            inner
                .free_physical_slots
                .try_reserve(removed_pages)
                .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
            for page in new_end_page..old_end_page {
                let mapping = inner
                    .mappings
                    .remove(address_space, page)
                    .expect("shrinking range was preflighted");
                let still_mapped = inner
                    .mappings
                    .mappings()
                    .any(|(_, _, candidate)| candidate.physical_slot == mapping.physical_slot);
                if !still_mapped {
                    inner.remove_page(mapping.physical_page, mapping.physical_slot);
                }
            }
            mutation.commit();
            return Ok(());
        }
        if new_end_page == old_end_page {
            return Ok(());
        }

        let backing_store = backing_store
            .ok_or_else(|| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
        let additional_pages = new_end_page - old_end_page;
        let capacity = usize::try_from(additional_pages)
            .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(capacity)
            .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
        let additional_slots = capacity.saturating_sub(inner.free_physical_slots.len());
        inner
            .physical_slots
            .try_reserve(additional_slots)
            .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
        let mut next_page_id = inner.next_page_id;
        for page in old_end_page..new_end_page {
            while inner
                .slots_by_id
                .contains_key(&GuestPhysicalPageId::new(next_page_id))
            {
                next_page_id = next_page_id
                    .checked_add(1)
                    .ok_or_else(|| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
            }
            let physical_page = GuestPhysicalPageId::new(next_page_id);
            next_page_id = next_page_id
                .checked_add(1)
                .ok_or_else(|| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
            let backing = CanonicalBackingPage::zeroed(
                &backing_store,
                physical_page,
                SYNTHETIC_PAGE_SIZE,
                ContentGeneration::new(1),
            )
            .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
            pending.push((page, physical_page, backing));
        }
        let mapping_generation = if pending.is_empty() {
            MappingGeneration::INITIAL
        } else {
            inner
                .take_mapping_generation()
                .ok_or_else(|| error(start, MemoryMappingErrorReason::GenerationExhausted))?
        };
        for (page, physical_page, backing) in pending {
            let slot = inner
                .push_page(physical_page, ExecutionPhysicalPage::Ram(backing))
                .expect("allocated physical identity is unique");
            let previous = inner.mappings.insert(
                address_space,
                page,
                ExecutionMapping {
                    physical_page,
                    physical_slot: slot,
                    mapping_generation,
                    permissions,
                    purpose,
                    attributes: MemoryAttributes::NONE,
                },
            );
            debug_assert!(previous.is_none());
        }
        inner.next_page_id = next_page_id;
        mutation.commit();
        Ok(())
    }

    fn set_permissions(
        &self,
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        size: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryProtectionError> {
        let page_size = SYNTHETIC_PAGE_SIZE as u64;
        let error = |address, reason| MemoryProtectionError {
            address_space,
            address,
            reason,
        };
        if size == 0 || !start.is_aligned_to(page_size) || !size.is_multiple_of(page_size) {
            return Err(error(start, MemoryProtectionErrorReason::InvalidRange));
        }
        if permissions.contains(MemoryPermissions::WRITE)
            && permissions.contains(MemoryPermissions::EXECUTE)
        {
            return Err(error(
                start,
                MemoryProtectionErrorReason::WritableExecutable,
            ));
        }
        let end = start
            .get()
            .checked_add(size)
            .ok_or_else(|| error(start, MemoryProtectionErrorReason::InvalidRange))?;
        let first_page = start.get() >> PAGE_SHIFT;
        let end_page = end >> PAGE_SHIFT;
        let mut mutation = self.begin_mapping_mutation();
        let mut inner = self.lock_inner();
        for page in first_page..end_page {
            let Some(mapping) = inner.mappings.get(address_space, page) else {
                return Err(error(
                    GuestVirtualAddress::new(page * page_size),
                    MemoryProtectionErrorReason::Unmapped,
                ));
            };
            if mapping
                .attributes
                .contains(MemoryAttributes::PERMISSION_LOCKED)
                && mapping.permissions != permissions
            {
                return Err(error(
                    GuestVirtualAddress::new(page * page_size),
                    MemoryProtectionErrorReason::PermissionLocked,
                ));
            }
        }
        let changed = (first_page..end_page).any(|page| {
            inner
                .mappings
                .get(address_space, page)
                .is_some_and(|mapping| mapping.permissions != permissions)
        });
        if !changed {
            return Ok(());
        }
        let mapping_generation = inner
            .take_mapping_generation()
            .ok_or_else(|| error(start, MemoryProtectionErrorReason::GenerationExhausted))?;
        for page in first_page..end_page {
            let mapping = inner
                .mappings
                .get_mut(address_space, page)
                .expect("protection range was preflighted");
            mapping.permissions = permissions;
            mapping.mapping_generation = mapping_generation;
        }
        mutation.commit();
        Ok(())
    }

    fn set_attributes(
        &self,
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        size: u64,
        mask: MemoryAttributes,
        value: MemoryAttributes,
    ) -> Result<(), MemoryProtectionError> {
        let page_size = SYNTHETIC_PAGE_SIZE as u64;
        let error = |address, reason| MemoryProtectionError {
            address_space,
            address,
            reason,
        };
        if size == 0
            || !start.is_aligned_to(page_size)
            || !size.is_multiple_of(page_size)
            || value.bits() & !mask.bits() != 0
        {
            return Err(error(start, MemoryProtectionErrorReason::InvalidRange));
        }
        let end = start
            .get()
            .checked_add(size)
            .ok_or_else(|| error(start, MemoryProtectionErrorReason::InvalidRange))?;
        let first_page = start.get() >> PAGE_SHIFT;
        let end_page = end >> PAGE_SHIFT;
        let mut mutation = self.begin_mapping_mutation();
        let mut inner = self.lock_inner();
        for page in first_page..end_page {
            if inner.mappings.get(address_space, page).is_none() {
                return Err(error(
                    GuestVirtualAddress::new(page * page_size),
                    MemoryProtectionErrorReason::Unmapped,
                ));
            }
        }
        let changed = (first_page..end_page).any(|page| {
            let mapping = inner
                .mappings
                .get(address_space, page)
                .expect("attribute range was preflighted");
            let bits = (mapping.attributes.bits() & !mask.bits()) | value.bits();
            mapping.attributes.bits() != bits
        });
        if !changed {
            return Ok(());
        }
        let mapping_generation = inner
            .take_mapping_generation()
            .ok_or_else(|| error(start, MemoryProtectionErrorReason::GenerationExhausted))?;
        for page in first_page..end_page {
            let mapping = inner
                .mappings
                .get_mut(address_space, page)
                .expect("attribute range was preflighted");
            let bits = (mapping.attributes.bits() & !mask.bits()) | value.bits();
            mapping.attributes = MemoryAttributes::from_bits(bits)
                .expect("existing, masked, and replacement attributes are bounded");
            mapping.mapping_generation = mapping_generation;
        }
        mutation.commit();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::memory::MemoryAccessSize;
    use nixe_memory::{
        CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint,
        DeviceVisibilityRequest, NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
        VisibilityState,
    };

    struct DeviceWriteback {
        bytes: Box<[u8]>,
    }

    impl VisibilityCoordinator for DeviceWriteback {
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
            Ok(self.bytes.clone())
        }
    }

    #[test]
    fn canonical_translation_retains_checked_page_spanning_segments() {
        let memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(7);
        memory
            .resize_zeroed_mapping(
                space,
                GuestVirtualAddress::new(0x1000),
                0,
                0x3000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();

        let range = memory
            .translate_canonical_range(
                space,
                GuestVirtualAddress::new(0x1800),
                0x1800,
                MemoryPermissions::WRITE,
            )
            .unwrap();
        assert_eq!(range.size(), 0x1800);
        assert_eq!(range.segments().len(), 2);
        assert_eq!(range.segments()[0].offset(), 0x800);
        assert_eq!(range.segments()[0].size(), 0x800);
        assert_eq!(range.segments()[1].offset(), 0);
        assert_eq!(range.segments()[1].size(), 0x1000);
        assert_eq!(
            range.segments()[0].permissions(),
            MemoryPermissions::READ_WRITE
        );
        assert_ne!(range.segments()[0].page(), range.segments()[1].page());
        assert_eq!(
            range.segments()[0].page().store(),
            range.segments()[1].page().store()
        );

        memory
            .set_permissions(
                space,
                GuestVirtualAddress::new(0x2000),
                0x1000,
                MemoryPermissions::READ,
            )
            .unwrap();
        assert_eq!(
            memory
                .translate_canonical_range(
                    space,
                    GuestVirtualAddress::new(0x1800),
                    0x1800,
                    MemoryPermissions::WRITE,
                )
                .unwrap_err(),
            CanonicalRangeTranslationError {
                address_space: space,
                address: GuestVirtualAddress::new(0x2000),
                reason: CanonicalRangeTranslationErrorReason::PermissionDenied,
            }
        );
    }

    #[test]
    fn mapped_ram_write_is_atomic_across_a_read_only_page() {
        let memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(9);
        memory
            .resize_zeroed_mapping(
                space,
                GuestVirtualAddress::new(0x1000),
                0,
                0x2000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();
        memory
            .set_permissions(
                space,
                GuestVirtualAddress::new(0x2000),
                0x1000,
                MemoryPermissions::READ,
            )
            .unwrap();
        assert!(!memory.write_mapped_ram(space, GuestVirtualAddress::new(0x1fff), &[0xaa, 0xbb],));
        assert_eq!(
            memory
                .read(
                    space,
                    GuestVirtualAddress::new(0x1fff),
                    MemoryAccess::normal(MemoryAccessSize::Byte),
                )
                .unwrap()
                .value,
            MemoryValue::U8(0),
        );
    }

    #[test]
    fn retained_translation_survives_cpu_unmap_and_memory_teardown() {
        let space = AddressSpaceId::new(9);
        let retained = {
            let memory = ExecutionMemory::new();
            memory
                .resize_zeroed_mapping(
                    space,
                    GuestVirtualAddress::new(0x4000),
                    0,
                    0x1000,
                    MemoryPermissions::READ_WRITE,
                    MemoryMappingPurpose::Heap,
                )
                .unwrap();
            memory
                .write(
                    space,
                    GuestVirtualAddress::new(0x4007),
                    MemoryAccess::normal(crate::memory::MemoryAccessSize::Byte),
                    MemoryValue::U8(0x5a),
                )
                .unwrap();
            let retained = memory
                .translate_canonical_range(
                    space,
                    GuestVirtualAddress::new(0x4000),
                    0x1000,
                    MemoryPermissions::READ,
                )
                .unwrap();
            memory
                .resize_zeroed_mapping(
                    space,
                    GuestVirtualAddress::new(0x4000),
                    0x1000,
                    0,
                    MemoryPermissions::READ_WRITE,
                    MemoryMappingPurpose::Heap,
                )
                .unwrap();
            assert_eq!(memory.physical_page_count(), 0);
            retained
        };

        assert_eq!(retained.size(), 0x1000);
        assert!(retained.segments()[0].content_is_current());
        assert_eq!(
            retained.segments()[0].content_generation(),
            ContentGeneration::new(2)
        );
    }

    #[test]
    fn cpu_read_reconciles_gpu_newer_backing_through_neutral_slow_path() {
        let memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(10);
        assert_eq!(
            memory.content_mutation_epoch(),
            ContentMutationEpoch::INITIAL
        );
        memory
            .resize_zeroed_mapping(
                space,
                GuestVirtualAddress::new(0x8000),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();
        let retained = memory
            .translate_canonical_range(
                space,
                GuestVirtualAddress::new(0x8000),
                0x1000,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let mut bytes = vec![0; 0x1000];
        bytes[9] = 0xa5;
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(DeviceWriteback {
            bytes: bytes.into_boxed_slice(),
        });
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(4),
            DeviceVisibilityPoint::new(11),
            DeviceVisibilityPoint::new(12),
        )
        .unwrap();
        retained
            .prepare_device_access(declaration, Arc::clone(&coordinator))
            .unwrap();
        retained
            .complete_device_write(declaration, Arc::clone(&coordinator))
            .unwrap();
        assert_eq!(
            memory.content_mutation_epoch(),
            ContentMutationEpoch::new(1)
        );
        assert!(matches!(
            retained.segments()[0].visibility_state(),
            VisibilityState::GpuNewer { .. }
        ));

        let result = memory
            .read(
                space,
                GuestVirtualAddress::new(0x8009),
                MemoryAccess::normal(crate::memory::MemoryAccessSize::Byte),
            )
            .unwrap();
        assert_eq!(result.value, MemoryValue::U8(0xa5));
        assert_eq!(
            memory.content_mutation_epoch(),
            ContentMutationEpoch::new(1),
            "downloading an already-published device write is not a second mutation"
        );
        assert_eq!(
            retained.segments()[0].visibility_state(),
            VisibilityState::Clean
        );

        let second_write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(4),
            DeviceVisibilityPoint::new(13),
            DeviceVisibilityPoint::new(14),
        )
        .unwrap();
        retained
            .prepare_device_access(second_write, Arc::clone(&coordinator))
            .unwrap();
        retained
            .complete_device_write(second_write, coordinator)
            .unwrap();
        assert_eq!(
            memory.content_mutation_epoch(),
            ContentMutationEpoch::new(2)
        );
        memory
            .write(
                space,
                GuestVirtualAddress::new(0x8009),
                MemoryAccess::normal(crate::memory::MemoryAccessSize::Byte),
                MemoryValue::U8(0x33),
            )
            .unwrap();
        assert_eq!(
            memory.content_mutation_epoch(),
            ContentMutationEpoch::new(3)
        );
        assert_eq!(
            retained.segments()[0].visibility_state(),
            VisibilityState::CpuNewer
        );
        assert_eq!(
            memory
                .read(
                    space,
                    GuestVirtualAddress::new(0x8009),
                    MemoryAccess::normal(crate::memory::MemoryAccessSize::Byte),
                )
                .unwrap()
                .value,
            MemoryValue::U8(0x33)
        );
    }

    #[test]
    fn gpu_write_invalidates_a_cpu_exclusive_reservation_before_store() {
        let memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(11);
        let address = GuestVirtualAddress::new(0x9000);
        let access = MemoryAccess::normal(crate::memory::MemoryAccessSize::Byte);
        memory
            .resize_zeroed_mapping(
                space,
                address,
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();
        let (_, reservation) = memory.load_exclusive(space, address, access).unwrap();
        let retained = memory
            .translate_canonical_range(space, address, 0x1000, MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut bytes = vec![0; 0x1000];
        bytes[0] = 0x5a;
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(DeviceWriteback {
            bytes: bytes.into_boxed_slice(),
        });
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(5),
            DeviceVisibilityPoint::new(20),
            DeviceVisibilityPoint::new(21),
        )
        .unwrap();
        retained
            .prepare_device_access(declaration, Arc::clone(&coordinator))
            .unwrap();
        retained
            .complete_device_write(declaration, coordinator)
            .unwrap();

        let (_, stored) = memory
            .store_exclusive(space, address, access, MemoryValue::U8(0xff), reservation)
            .unwrap();
        assert!(!stored);
        assert_eq!(
            memory.read(space, address, access).unwrap().value,
            MemoryValue::U8(0x5a)
        );
    }

    #[test]
    fn sparse_page_table_allocates_only_populated_leaves() {
        let mut memory = ExecutionMemory::new();
        let low = GuestPhysicalPageId::new(1);
        let high = GuestPhysicalPageId::new(2);
        assert!(memory.add_ram_page(low));
        assert!(memory.add_ram_page(high));
        assert!(memory.map_page(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x1000),
            low,
            MemoryPermissions::READ,
        ));
        assert!(memory.map_page(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0xffff_ffff_ffff_f000),
            high,
            MemoryPermissions::READ,
        ));

        let inner = memory.inner_mut();
        assert_eq!(inner.mappings.leaves.len(), 2);
        assert_eq!(inner.mappings.mappings().count(), 2);
    }

    #[test]
    fn released_physical_slots_are_reused_without_unbounded_growth() {
        let memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(1);
        let base = GuestVirtualAddress::new(0x20_0000);
        let size = (SYNTHETIC_PAGE_SIZE * 4) as u64;

        memory
            .resize_zeroed_mapping(
                space,
                base,
                0,
                size,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();
        memory
            .resize_zeroed_mapping(
                space,
                base,
                size,
                0,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();
        memory
            .resize_zeroed_mapping(
                space,
                base,
                0,
                size,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();

        let inner = memory.lock_inner();
        assert_eq!(inner.physical_slots.len(), 4);
        assert_eq!(inner.slots_by_id.len(), 4);
        assert!(inner.free_physical_slots.is_empty());
    }

    #[test]
    fn identity_exhaustion_does_not_publish_partial_installation() {
        let mut memory = ExecutionMemory::new();
        memory.inner_mut().next_page_id = u64::MAX;
        let bytes = [0x5a; SYNTHETIC_PAGE_SIZE];
        let address = GuestVirtualAddress::new(0x1000);

        let error = memory
            .install_ram_pages_atomic(
                AddressSpaceId::new(1),
                &[SyntheticRamPage {
                    virtual_address: address,
                    bytes: &bytes,
                    permissions: MemoryPermissions::READ_EXECUTE,
                }],
            )
            .unwrap_err();

        assert_eq!(error.stage, SyntheticInstallStage::Allocation);
        assert_eq!(memory.physical_page_count(), 0);
        assert!(
            memory
                .mapping_info(AddressSpaceId::new(1), address)
                .is_none()
        );
    }
}
