//! Production process-memory storage.
//!
//! [`ExecutionMemory`] deliberately does not reuse [`super::SyntheticMemory`].
//! The synthetic backend favors deterministic fault injection and simple
//! observability. This backend instead resolves a virtual page through one
//! sparse radix leaf and then indexes a stable physical-page slot directly.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter},
    marker::PhantomData,
    sync::atomic::AtomicU64,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use nixe_memory::{
    AddressSpaceId, CanonicalBackingPage, CanonicalBackingRange, CanonicalBackingSegment,
    CanonicalBackingStore, CanonicalPageError, CanonicalRangeTranslationError,
    CanonicalRangeTranslationErrorReason, CanonicalRangeTranslator, CanonicalWriteBatch,
    ContentGeneration, CpuMemoryBackend, DirectAddressSpaceView, DirectArena, DirectBackendPolicy,
    DirectMapRequest, DirectProtectRequest, DirectProtection, ExecutionGate, ExecutionSharedGuard,
    ExecutionTransitionGuard, GuestPhysicalPageId, GuestVirtualAddress, HostMappedBacking,
    MappingGeneration, MemoryInvalidation, MemoryInvalidationCursor, MemoryInvalidationError,
    MemoryInvalidationKind, MemoryInvalidationLog, MemoryInvalidationOrigin,
    MemoryInvalidationSource,
};

use crate::{
    error::{InstructionFetchFault, InstructionFetchFaultReason},
    exclusive::ExclusiveReservation,
};

use super::common::{
    MappingState, PAGE_SIZE, PageRange, ResolvedDataAccess, allocate_page_id,
    coalesce_mapped_pages, install_error, masked_attributes, memory_query_result, page_address,
    page_offset, resolve_data_access, take_mapping_generation, validate_install_request,
    virtual_page, writable_executable,
};
use super::{
    AtomicMemoryResult, AtomicRmwKind, CodeDependencies, CodePageDependency, CodePageSpan,
    CpuMemory, DataAccessFault, DataAccessFaultReason, DataAccessKind, DataReadResult,
    DataWriteResult, DirectFaultResolution, FetchedCode, InstructionMemory, MemoryAccess,
    MemoryAccessClass, MemoryAliasError, MemoryAliasErrorReason, MemoryAliasRequest,
    MemoryAlignment, MemoryAttributes, MemoryMappingError, MemoryMappingErrorReason,
    MemoryMappingProperties, MemoryMappingPurpose, MemoryPermissions, MemoryProtectionError,
    MemoryProtectionErrorReason, MemoryQueryResult, MemoryRegionKind, MemoryValue, ProcessMemory,
    SYNTHETIC_PAGE_SIZE, SyntheticInstallError, SyntheticInstallStage, SyntheticMappingInfo,
    SyntheticMmio, SyntheticRamPage,
};

const LEAF_BITS: u32 = 9;
const LEAF_ENTRY_COUNT: usize = 1 << LEAF_BITS;
const LEAF_INDEX_MASK: u64 = (LEAF_ENTRY_COUNT as u64) - 1;

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

struct ExecutionPhysicalSlot {
    page: ExecutionPhysicalPage,
    // Mapping-derived metadata is maintained transactionally with the page
    // table. Guest data accesses can therefore classify aliases in O(1)
    // without walking the process address space.
    mapping_count: usize,
    executable_content_mapping_count: usize,
    aliases: BTreeSet<(AddressSpaceId, u64)>,
}

enum ExecutionBackendBinding {
    Checked,
    LinuxDirect {
        arena: DirectArena,
        address_space_size: usize,
    },
}

#[derive(Default)]
struct ExecutionMemoryInner {
    // Every published mapping's slot contains a page and its physical ID maps
    // back to that same slot. Aliases intentionally repeat both values. A free
    // slot is absent from all mappings and from `slots_by_id`.
    mappings: ExecutionPageTable,
    physical_slots: Vec<Option<ExecutionPhysicalSlot>>,
    free_physical_slots: Vec<usize>,
    slots_by_id: BTreeMap<GuestPhysicalPageId, usize>,
    backends: BTreeMap<AddressSpaceId, ExecutionBackendBinding>,
    backend_reasons: BTreeMap<AddressSpaceId, Box<str>>,
    direct_failure: Option<Box<str>>,
    next_page_id: u64,
    next_mapping_generation: Option<MappingGeneration>,
}

impl ExecutionMemoryInner {
    fn page(&self, slot: usize) -> Option<&ExecutionPhysicalPage> {
        Some(&self.physical_slots.get(slot)?.as_ref()?.page)
    }

    fn page_mut(&mut self, slot: usize) -> Option<&mut ExecutionPhysicalPage> {
        Some(&mut self.physical_slots.get_mut(slot)?.as_mut()?.page)
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
            *destination = Some(ExecutionPhysicalSlot {
                page,
                mapping_count: 0,
                executable_content_mapping_count: 0,
                aliases: BTreeSet::new(),
            });
            slot
        } else {
            let slot = self.physical_slots.len();
            self.physical_slots.push(Some(ExecutionPhysicalSlot {
                page,
                mapping_count: 0,
                executable_content_mapping_count: 0,
                aliases: BTreeSet::new(),
            }));
            slot
        };
        self.slots_by_id.insert(id, slot);
        Some(slot)
    }

    fn remove_page(&mut self, id: GuestPhysicalPageId, slot: usize) {
        let physical_slot = self
            .physical_slots
            .get(slot)
            .and_then(Option::as_ref)
            .expect("removed physical slot exists");
        assert_eq!(physical_slot.mapping_count, 0);
        assert_eq!(physical_slot.executable_content_mapping_count, 0);
        assert!(physical_slot.aliases.is_empty());
        let removed_id = self.slots_by_id.remove(&id);
        let removed_page = self.physical_slots.get_mut(slot).and_then(Option::take);
        debug_assert_eq!(removed_id, Some(slot));
        debug_assert!(removed_page.is_some());
        self.free_physical_slots.push(slot);
    }

    fn insert_mapping(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
        mapping: ExecutionMapping,
    ) {
        self.insert_mapping_unpublished(address_space, virtual_page, mapping);
        self.publish_direct_mapping(address_space, virtual_page);
    }

    fn insert_mapping_unpublished(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
        mapping: ExecutionMapping,
    ) {
        assert!(self.mappings.get(address_space, virtual_page).is_none());
        let previous = self.mappings.insert(address_space, virtual_page, mapping);
        debug_assert!(previous.is_none());
        self.register_mapping(address_space, virtual_page, mapping);
    }

    fn remove_mapping_unpublished(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
    ) -> Option<ExecutionMapping> {
        let mapping = self.mappings.remove(address_space, virtual_page)?;
        self.unregister_mapping(address_space, virtual_page, mapping);
        Some(mapping)
    }

    fn set_mapping_purpose(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
        purpose: MemoryMappingPurpose,
        mapping_generation: MappingGeneration,
    ) {
        let (physical_slot, was_executable_content, is_executable_content) = {
            let mapping = self
                .mappings
                .get_mut(address_space, virtual_page)
                .expect("mapping purpose range was preflighted");
            let was_executable_content = mapping.observes_executable_content();
            mapping.purpose = purpose;
            mapping.mapping_generation = mapping_generation;
            (
                mapping.physical_slot,
                was_executable_content,
                mapping.observes_executable_content(),
            )
        };
        self.update_executable_content_mapping_count(
            physical_slot,
            was_executable_content,
            is_executable_content,
        );
    }

    fn set_mapping_permissions(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
        permissions: MemoryPermissions,
        mapping_generation: MappingGeneration,
    ) {
        let (physical_slot, was_executable_content, is_executable_content) = {
            let mapping = self
                .mappings
                .get_mut(address_space, virtual_page)
                .expect("mapping protection range was preflighted");
            let was_executable_content = mapping.observes_executable_content();
            mapping.permissions = permissions;
            mapping.mapping_generation = mapping_generation;
            (
                mapping.physical_slot,
                was_executable_content,
                mapping.observes_executable_content(),
            )
        };
        self.update_executable_content_mapping_count(
            physical_slot,
            was_executable_content,
            is_executable_content,
        );
    }

    fn set_mapping_properties(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
        properties: MemoryMappingProperties,
        mapping_generation: MappingGeneration,
    ) {
        let (physical_slot, was_executable_content, is_executable_content) = {
            let mapping = self
                .mappings
                .get_mut(address_space, virtual_page)
                .expect("mapping-property range was preflighted");
            let was_executable_content = mapping.observes_executable_content();
            mapping.permissions = properties.permissions;
            mapping.purpose = properties.purpose;
            mapping.attributes = properties.attributes;
            mapping.mapping_generation = mapping_generation;
            (
                mapping.physical_slot,
                was_executable_content,
                mapping.observes_executable_content(),
            )
        };
        self.update_executable_content_mapping_count(
            physical_slot,
            was_executable_content,
            is_executable_content,
        );
    }

    fn executable_content_page(&self, physical_slot: usize) -> Option<GuestPhysicalPageId> {
        let slot = self.physical_slots.get(physical_slot)?.as_ref()?;
        if slot.executable_content_mapping_count == 0 {
            return None;
        }
        match &slot.page {
            ExecutionPhysicalPage::Ram(page) => Some(page.identity().page()),
            ExecutionPhysicalPage::Mmio(_) => None,
        }
    }

    fn mapping_count(&self, physical_slot: usize) -> usize {
        self.physical_slots
            .get(physical_slot)
            .and_then(Option::as_ref)
            .map_or(0, |slot| slot.mapping_count)
    }

    fn register_mapping(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
        mapping: ExecutionMapping,
    ) {
        let slot = self
            .physical_slots
            .get_mut(mapping.physical_slot)
            .and_then(Option::as_mut)
            .expect("mapping references an owned physical slot");
        slot.mapping_count = slot
            .mapping_count
            .checked_add(1)
            .expect("physical mapping count is bounded by guest mappings");
        assert!(slot.aliases.insert((address_space, virtual_page)));
        debug_assert_eq!(slot.aliases.len(), slot.mapping_count);
        if mapping.observes_executable_content() {
            slot.executable_content_mapping_count = slot
                .executable_content_mapping_count
                .checked_add(1)
                .expect("executable mapping count is bounded by guest mappings");
        }
    }

    fn unregister_mapping(
        &mut self,
        address_space: AddressSpaceId,
        virtual_page: u64,
        mapping: ExecutionMapping,
    ) {
        let slot = self
            .physical_slots
            .get_mut(mapping.physical_slot)
            .and_then(Option::as_mut)
            .expect("mapping references an owned physical slot");
        slot.mapping_count = slot
            .mapping_count
            .checked_sub(1)
            .expect("physical mapping count tracks every published mapping");
        assert!(slot.aliases.remove(&(address_space, virtual_page)));
        debug_assert_eq!(slot.aliases.len(), slot.mapping_count);
        if mapping.observes_executable_content() {
            slot.executable_content_mapping_count = slot
                .executable_content_mapping_count
                .checked_sub(1)
                .expect("executable mapping count tracks every executable alias");
        }
    }

    fn update_executable_content_mapping_count(
        &mut self,
        physical_slot: usize,
        was_executable_content: bool,
        is_executable_content: bool,
    ) {
        if was_executable_content == is_executable_content {
            return;
        }
        let slot = self
            .physical_slots
            .get_mut(physical_slot)
            .and_then(Option::as_mut)
            .expect("mapping references an owned physical slot");
        if is_executable_content {
            slot.executable_content_mapping_count = slot
                .executable_content_mapping_count
                .checked_add(1)
                .expect("executable mapping count is bounded by guest mappings");
        } else {
            slot.executable_content_mapping_count = slot
                .executable_content_mapping_count
                .checked_sub(1)
                .expect("executable mapping count tracks every executable alias");
        }
    }

    fn publish_direct_mapping(&mut self, address_space: AddressSpaceId, virtual_page: u64) {
        if self.direct_failure.is_some() {
            return;
        }
        let Some(ExecutionBackendBinding::LinuxDirect { arena, .. }) =
            self.backends.get(&address_space)
        else {
            return;
        };
        let arena = arena.clone();
        let Some(mapping) = self.mappings.get(address_space, virtual_page) else {
            return;
        };
        let Some(ExecutionPhysicalPage::Ram(backing)) = self.page(mapping.physical_slot) else {
            return;
        };
        let backing = backing.clone();
        let guest_address = page_address(virtual_page).get();
        let result = (|| {
            let host = backing
                .direct_backing()
                .map_err(|error| CpuMemoryBackendError::new(error.to_string()))?;
            let maximum = maximum_direct_protection(mapping);
            let protection = effective_direct_protection(maximum, &backing);
            arena
                .reconcile_page(
                    guest_address,
                    Some(DirectMapRequest {
                        guest_address,
                        backing: &host,
                        protection,
                    }),
                )
                .map_err(|error| CpuMemoryBackendError::new(error.to_string()))?;
            backing
                .register_direct_alias(&arena, guest_address, maximum)
                .map_err(|error| CpuMemoryBackendError::new(error.to_string()))
        })();
        if let Err(error) = result {
            self.direct_failure = Some(error.to_string().into_boxed_str());
        }
    }

    fn publish_direct_mapping_range(
        &mut self,
        address_space: AddressSpaceId,
        first_page: u64,
        end_page: u64,
    ) {
        if self.direct_failure.is_some() || first_page == end_page {
            return;
        }
        let Some(ExecutionBackendBinding::LinuxDirect { arena, .. }) =
            self.backends.get(&address_space)
        else {
            return;
        };
        let arena = arena.clone();
        let result = (|| {
            let capacity = usize::try_from(end_page - first_page)
                .map_err(|_| CpuMemoryBackendError::new("direct map range is too large"))?;
            let mut desired = Vec::new();
            desired
                .try_reserve_exact(capacity)
                .map_err(|_| CpuMemoryBackendError::new("direct map range allocation failed"))?;
            for virtual_page in first_page..end_page {
                let mapping = self
                    .mappings
                    .get(address_space, virtual_page)
                    .expect("published direct range retains every canonical mapping");
                let Some(ExecutionPhysicalPage::Ram(backing)) = self.page(mapping.physical_slot)
                else {
                    continue;
                };
                let backing = backing.clone();
                let host = backing
                    .direct_backing()
                    .map_err(|error| CpuMemoryBackendError::new(error.to_string()))?;
                let maximum = maximum_direct_protection(mapping);
                let protection = effective_direct_protection(maximum, &backing);
                desired.push((
                    page_address(virtual_page).get(),
                    backing,
                    host,
                    maximum,
                    protection,
                ));
            }
            let mut requests = Vec::new();
            requests
                .try_reserve_exact(desired.len())
                .map_err(|_| CpuMemoryBackendError::new("direct map request allocation failed"))?;
            requests.extend(
                desired
                    .iter()
                    .map(|(guest_address, _, host, _, protection)| DirectMapRequest {
                        guest_address: *guest_address,
                        backing: host,
                        protection: *protection,
                    }),
            );
            arena
                .reconcile_mapped_pages(&requests)
                .map_err(|error| CpuMemoryBackendError::new(error.to_string()))?;
            for (guest_address, backing, _, maximum, _) in desired {
                backing
                    .register_direct_alias(&arena, guest_address, maximum)
                    .map_err(|error| CpuMemoryBackendError::new(error.to_string()))?;
            }
            Ok::<(), CpuMemoryBackendError>(())
        })();
        if let Err(error) = result {
            self.direct_failure = Some(error.to_string().into_boxed_str());
        }
    }

    fn revoke_direct_mapping_range(
        &mut self,
        address_space: AddressSpaceId,
        first_page: u64,
        end_page: u64,
    ) {
        if self.direct_failure.is_some() || first_page == end_page {
            return;
        }
        let Some(ExecutionBackendBinding::LinuxDirect { arena, .. }) =
            self.backends.get(&address_space)
        else {
            return;
        };
        let arena = arena.clone();
        for virtual_page in first_page..end_page {
            let guest_address = page_address(virtual_page).get();
            if let Some(mapping) = self.mappings.get(address_space, virtual_page)
                && let Some(ExecutionPhysicalPage::Ram(backing)) = self.page(mapping.physical_slot)
            {
                backing.unregister_direct_alias(&arena, guest_address);
            }
        }
        let page_count = end_page - first_page;
        let result = usize::try_from(page_count)
            .ok()
            .and_then(|count| count.checked_mul(SYNTHETIC_PAGE_SIZE))
            .ok_or_else(|| CpuMemoryBackendError::new("direct revoke range is too large"))
            .and_then(|size| {
                arena
                    .replace_with_none(&[DirectProtectRequest {
                        guest_address: page_address(first_page).get(),
                        size,
                        protection: DirectProtection::None,
                    }])
                    .map_err(|error| CpuMemoryBackendError::new(error.to_string()))
            });
        if let Err(error) = result {
            self.direct_failure = Some(error.to_string().into_boxed_str());
        }
    }

    fn mapping_at(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Option<ExecutionMapping> {
        self.mappings.get(address_space, virtual_page(address))
    }

    fn mapping_state(
        &self,
        address_space: AddressSpaceId,
        virtual_page: u64,
    ) -> Option<MappingState> {
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

impl ExecutionMapping {
    fn properties(self) -> MemoryMappingProperties {
        MemoryMappingProperties::new(self.permissions, self.purpose, self.attributes)
    }

    fn observes_executable_content(self) -> bool {
        self.permissions.contains(MemoryPermissions::EXECUTE) || self.purpose.is_code()
    }
}

fn synchronize_direct_backend(
    inner: &mut ExecutionMemoryInner,
    address_space: AddressSpaceId,
) -> Result<(), CpuMemoryBackendError> {
    let (arena, address_space_size) = match inner.backends.get(&address_space) {
        Some(ExecutionBackendBinding::LinuxDirect {
            arena,
            address_space_size,
        }) => (arena.clone(), *address_space_size),
        Some(ExecutionBackendBinding::Checked) => return Ok(()),
        None => {
            return Err(CpuMemoryBackendError::new(
                "direct synchronization requested for an unbound address space",
            ));
        }
    };
    let mut desired = Vec::<(
        u64,
        CanonicalBackingPage,
        HostMappedBacking,
        DirectProtection,
    )>::new();
    for (mapped_space, virtual_page, mapping) in inner.mappings.mappings() {
        if mapped_space != address_space {
            continue;
        }
        let guest_address = virtual_page
            .checked_mul(SYNTHETIC_PAGE_SIZE as u64)
            .ok_or_else(|| CpuMemoryBackendError::new("direct guest page address overflows"))?;
        let guest_end = usize::try_from(guest_address)
            .ok()
            .and_then(|start| start.checked_add(SYNTHETIC_PAGE_SIZE));
        if guest_end.is_none_or(|end| end > address_space_size) {
            return Err(CpuMemoryBackendError::new(
                "canonical mapping lies outside its direct address-space arena",
            ));
        }
        let Some(ExecutionPhysicalPage::Ram(backing)) = inner.page(mapping.physical_slot) else {
            continue;
        };
        let host = backing
            .direct_backing()
            .map_err(|error| CpuMemoryBackendError::new(error.to_string()))?;
        desired.push((
            guest_address,
            backing.clone(),
            host,
            effective_direct_protection(maximum_direct_protection(mapping), backing),
        ));
    }
    desired.sort_unstable_by_key(|(guest, _, _, _)| *guest);
    let requests = desired
        .iter()
        .map(|(guest_address, _, backing, protection)| DirectMapRequest {
            guest_address: *guest_address,
            backing,
            protection: *protection,
        })
        .collect::<Vec<_>>();
    arena
        .reconcile_pages(&requests)
        .map_err(|error| CpuMemoryBackendError::new(error.to_string()))?;
    for (guest_address, page, _, _) in desired {
        let mapping = inner
            .mappings
            .get(address_space, guest_address / SYNTHETIC_PAGE_SIZE as u64)
            .expect("direct binding desired pages retain their mappings");
        page.register_direct_alias(&arena, guest_address, maximum_direct_protection(mapping))
            .map_err(|error| CpuMemoryBackendError::new(error.to_string()))?;
    }
    Ok(())
}

fn maximum_direct_protection(mapping: ExecutionMapping) -> DirectProtection {
    if !mapping.permissions.contains(MemoryPermissions::READ) {
        return DirectProtection::None;
    }
    if mapping.permissions.contains(MemoryPermissions::WRITE) {
        DirectProtection::ReadWrite
    } else {
        DirectProtection::Read
    }
}

fn effective_direct_protection(
    maximum: DirectProtection,
    backing: &CanonicalBackingPage,
) -> DirectProtection {
    if !matches!(
        backing.visibility_state(),
        nixe_memory::VisibilityState::Clean | nixe_memory::VisibilityState::CpuNewer
    ) {
        return DirectProtection::None;
    }
    match maximum {
        DirectProtection::ReadWrite if !backing.direct_write_is_armed() => DirectProtection::Read,
        protection => protection,
    }
}

/// Process-memory backend used by normal guest execution.
///
/// The backend owns production-specific sparse page tables and physical-page
/// slots. It shares public semantic types with [`super::SyntheticMemory`], but
/// neither its storage nor any instruction/data hot path delegates to it.
///
/// Semantic accesses are serialized by one process-memory transaction lock.
/// One process-memory transaction makes cross-page validation, canonical-page
/// generation changes, MMIO callbacks, and mapping lookup indivisible while
/// permitting the memory object to be shared by concurrent vCPU workers.
pub struct ExecutionMemory {
    backing_store: Option<CanonicalBackingStore>,
    invalidations: Arc<MemoryInvalidationLog>,
    inner: Mutex<ExecutionMemoryInner>,
    execution_gate: ExecutionGate,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MappingEpoch(u64);

impl MappingEpoch {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// RAII proof that one executor may use mappings from the recorded epoch.
/// Mapping mutation waits for every such lease to be released at a safepoint.
pub struct ExecutionMemoryLease<'memory> {
    _shared: ExecutionSharedGuard,
    _memory: PhantomData<&'memory ExecutionMemory>,
    gate_identity: usize,
    epoch: MappingEpoch,
}

impl ExecutionMemoryLease<'_> {
    #[must_use]
    pub const fn epoch(&self) -> MappingEpoch {
        self.epoch
    }

    /// Proves that this live shared guard belongs to `memory`.
    #[must_use]
    pub fn authorizes(&self, memory: &dyn CpuMemory) -> bool {
        memory.execution_gate_identity() == Some(self.gate_identity)
    }
}

struct MappingMutationGuard<'a> {
    transition: ExecutionTransitionGuard<'a>,
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
            self.transition.commit();
        }
    }
}

/// Failure to construct or publish one immutable process memory backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuMemoryBackendError(Box<str>);

impl CpuMemoryBackendError {
    fn new(detail: impl Into<Box<str>>) -> Self {
        Self(detail.into())
    }
}

impl Display for CpuMemoryBackendError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CpuMemoryBackendError {}

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
        let execution_gate = ExecutionGate::new();
        Self {
            backing_store: CanonicalBackingStore::allocate_with_execution_gate(
                execution_gate.clone(),
            )
            .ok(),
            invalidations: Arc::new(MemoryInvalidationLog::default()),
            inner: Mutex::new(inner),
            execution_gate,
        }
    }

    fn lock_inner(&self) -> MutexGuard<'_, ExecutionMemoryInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn inner_mut(&mut self) -> &mut ExecutionMemoryInner {
        self.inner.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    /// Acquires one mapping-stable execution lease for a bounded engine slice.
    pub fn acquire_execution_lease(&self) -> ExecutionMemoryLease<'_> {
        let shared = self.execution_gate.acquire_shared();
        let epoch = MappingEpoch(shared.epoch());
        ExecutionMemoryLease {
            _shared: shared,
            _memory: PhantomData,
            gate_identity: self.execution_gate.identity(),
            epoch,
        }
    }

    /// Connects external ownership transitions to the runtime's bounded CPU
    /// safepoint request. The callback is cold and is invoked only when an
    /// exclusive transition finds an active execution slice.
    pub fn set_transition_notifier(&self, notifier: Option<Arc<dyn Fn() + Send + Sync>>) {
        self.execution_gate.set_transition_notifier(notifier);
    }

    /// Returns the mapping epoch visible to newly acquired execution leases.
    #[must_use]
    pub fn mapping_epoch(&self) -> MappingEpoch {
        MappingEpoch(self.execution_gate.epoch())
    }

    /// Reports whether a mapping mutation has closed admission to new engine
    /// slices and is waiting for, or currently owns, quiescence.
    #[must_use]
    pub fn mapping_mutation_pending(&self) -> bool {
        self.execution_gate.transition_pending()
    }

    fn begin_mapping_mutation(&self) -> MappingMutationGuard<'_> {
        MappingMutationGuard {
            transition: self.execution_gate.acquire_exclusive(),
            committed: false,
        }
    }

    /// Selects and eagerly publishes the immutable backend for one address
    /// space before an engine is bound.
    pub fn bind_cpu_memory_backend(
        &mut self,
        address_space: AddressSpaceId,
        address_space_size: u64,
        policy: DirectBackendPolicy,
    ) -> Result<CpuMemoryBackend, CpuMemoryBackendError> {
        let size = usize::try_from(address_space_size).map_err(|_| {
            CpuMemoryBackendError::new("guest address-space size exceeds host usize")
        })?;
        if size == 0 || !size.is_multiple_of(SYNTHETIC_PAGE_SIZE) {
            return Err(CpuMemoryBackendError::new(
                "guest address-space size is not a nonzero page multiple",
            ));
        }
        let inner = self.inner_mut();
        if inner.backends.contains_key(&address_space) {
            return Err(CpuMemoryBackendError::new(
                "CPU memory backend is already bound for this address space",
            ));
        }
        if matches!(policy, DirectBackendPolicy::Disabled) {
            inner
                .backends
                .insert(address_space, ExecutionBackendBinding::Checked);
            inner.backend_reasons.insert(
                address_space,
                "checked backend selected by Disabled policy".into(),
            );
            return Ok(CpuMemoryBackend::Checked);
        }
        let arena = match DirectArena::new(size) {
            Ok(arena) => arena,
            Err(error)
                if matches!(policy, DirectBackendPolicy::Preferred) && error.is_unsupported() =>
            {
                inner
                    .backends
                    .insert(address_space, ExecutionBackendBinding::Checked);
                inner.backend_reasons.insert(
                    address_space,
                    format!("checked fallback because LinuxDirect is unsupported: {error}")
                        .into_boxed_str(),
                );
                return Ok(CpuMemoryBackend::Checked);
            }
            Err(error) => return Err(CpuMemoryBackendError::new(error.to_string())),
        };
        inner.backends.insert(
            address_space,
            ExecutionBackendBinding::LinuxDirect {
                arena,
                address_space_size: size,
            },
        );
        if let Err(error) = synchronize_direct_backend(inner, address_space) {
            inner.backends.remove(&address_space);
            return Err(error);
        }
        inner.backend_reasons.insert(
            address_space,
            "LinuxDirect selected after host capability validation".into(),
        );
        Ok(CpuMemoryBackend::LinuxDirect)
    }

    /// Returns the immutable backend selected before engine publication.
    #[must_use]
    pub fn cpu_memory_backend(&self, address_space: AddressSpaceId) -> Option<CpuMemoryBackend> {
        self.lock_inner()
            .backends
            .get(&address_space)
            .map(|binding| match binding {
                ExecutionBackendBinding::Checked => CpuMemoryBackend::Checked,
                ExecutionBackendBinding::LinuxDirect { .. } => CpuMemoryBackend::LinuxDirect,
            })
    }

    /// Returns the cold construction diagnostic for the immutable backend.
    pub fn cpu_memory_backend_reason(&self, address_space: AddressSpaceId) -> Option<Box<str>> {
        self.lock_inner()
            .backend_reasons
            .get(&address_space)
            .cloned()
    }

    /// Returns the direct pointer view selected for generated code or stubs.
    #[must_use]
    pub fn direct_address_space_view(
        &self,
        address_space: AddressSpaceId,
    ) -> Option<DirectAddressSpaceView> {
        match self.lock_inner().backends.get(&address_space)? {
            ExecutionBackendBinding::Checked => None,
            ExecutionBackendBinding::LinuxDirect { arena, .. } => Some(arena.view()),
        }
    }

    #[must_use]
    pub fn direct_protection_at(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Option<DirectProtection> {
        match self.lock_inner().backends.get(&address_space)? {
            ExecutionBackendBinding::Checked => None,
            ExecutionBackendBinding::LinuxDirect { arena, .. } => {
                arena.protection_at(address.get())
            }
        }
    }

    /// Returns a deterministic backend publication failure, if one poisoned
    /// the arena during a later canonical mapping transition.
    #[must_use]
    pub fn direct_backend_failure(&self) -> Option<Box<str>> {
        self.lock_inner().direct_failure.clone()
    }

    /// Atomically installs initialized RAM pages.
    pub fn install_ram_pages_atomic(
        &mut self,
        address_space: AddressSpaceId,
        requests: &[SyntheticRamPage<'_>],
    ) -> Result<(), SyntheticInstallError> {
        let backing_store = self.backing_store.clone();
        let invalidations = &self.invalidations;
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        let mut virtual_pages = Vec::with_capacity(requests.len());
        let mut unique_virtual_pages = BTreeSet::new();
        for request in requests {
            let virtual_page = validate_install_request(*request, &mut unique_virtual_pages)?;
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
            let physical_page = allocate_page_id(&mut next_page_id, |page| {
                inner.slots_by_id.contains_key(&page)
            })
            .ok_or_else(|| {
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
        let mut invalidation_kinds = Vec::new();
        invalidation_kinds
            .try_reserve_exact(virtual_pages.len())
            .map_err(|_| {
                install_error(
                    SyntheticInstallStage::Publication,
                    requests.first().map(|request| request.virtual_address),
                    "memory invalidation allocation failed",
                )
            })?;
        for virtual_page in &virtual_pages {
            invalidation_kinds.push(MemoryInvalidationKind::Mapping {
                address_space,
                start: page_address(*virtual_page),
                size: PAGE_SIZE,
            });
        }
        let invalidation = invalidations
            .reserve_many(&invalidation_kinds)
            .map_err(|reason| {
                install_error(
                    SyntheticInstallStage::Publication,
                    requests.first().map(|request| request.virtual_address),
                    reason.to_string(),
                )
            })?;
        let mapping_generation = if pending.is_empty() {
            MappingGeneration::INITIAL
        } else {
            take_mapping_generation(&mut inner.next_mapping_generation).ok_or_else(|| {
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
            inner.insert_mapping(
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
        }
        inner.next_page_id = next_page_id;
        invalidation.commit();
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
        let Some(range) = PageRange::new(start, size).filter(|range| !range.is_empty()) else {
            return false;
        };
        let invalidations = &self.invalidations;
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        if !(range.first..range.end).all(|page| inner.mappings.get(address_space, page).is_some()) {
            return false;
        }
        if (range.first..range.end).all(|page| {
            inner
                .mappings
                .get(address_space, page)
                .is_some_and(|mapping| mapping.purpose == purpose)
        }) {
            return true;
        }
        let Ok(invalidation) = invalidations.reserve(MemoryInvalidationKind::Mapping {
            address_space,
            start,
            size,
        }) else {
            return false;
        };
        let Some(mapping_generation) = take_mapping_generation(&mut inner.next_mapping_generation)
        else {
            return false;
        };
        for page in range.first..range.end {
            inner.set_mapping_purpose(address_space, page, purpose, mapping_generation);
        }
        inner.publish_direct_mapping_range(address_space, range.first, range.end);
        invalidation.commit();
        true
    }

    /// Returns the number of physical pages owned by this backend.
    #[must_use]
    pub fn physical_page_count(&self) -> usize {
        self.lock_inner().slots_by_id.len()
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
        let invalidations = &self.invalidations;
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
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
        let invalidation = match inner.executable_content_page(slot) {
            Some(first) => match invalidations.reserve_with_origin(
                MemoryInvalidationKind::ExecutableContent {
                    first,
                    second: None,
                },
                MemoryInvalidationOrigin::HostWrite,
            ) {
                Ok(invalidation) => Some(invalidation),
                Err(_) => return false,
            },
            None => None,
        };
        let written = backing
            .write_preflighted(offset, bytes, generation, next_generation)
            .is_ok();
        if written && let Some(invalidation) = invalidation {
            invalidation.commit();
        }
        written
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
        self.overwrite_bytes_checked(address_space, address, bytes)
            .is_ok()
    }

    fn overwrite_bytes_checked(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        bytes: &[u8],
    ) -> Result<(), DataAccessFault> {
        let size = u64::try_from(bytes.len()).map_err(|_| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::AddressOverflow,
            )
        })?;
        let end = address.get().checked_add(size).ok_or_else(|| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::AddressOverflow,
            )
        })?;
        let inner = self.lock_inner();
        let mut cursor = address.get();
        let mut pending_generations = BTreeMap::new();
        while cursor < end {
            let virtual_address = GuestVirtualAddress::new(cursor);
            let mapping = inner
                .mapping_at(address_space, virtual_address)
                .ok_or_else(|| {
                    DataAccessFault::new(
                        address_space,
                        virtual_address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::Unmapped,
                    )
                })?;
            let Some(ExecutionPhysicalPage::Ram(backing)) = inner.page(mapping.physical_slot)
            else {
                return Err(DataAccessFault::new(
                    address_space,
                    virtual_address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::Device("bulk guest-memory writes require RAM".into()),
                ));
            };
            if let std::collections::btree_map::Entry::Vacant(entry) =
                pending_generations.entry(mapping.physical_slot)
            {
                backing.prepare_write().map_err(|reason| {
                    DataAccessFault::new(
                        address_space,
                        virtual_address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::HostBacking(reason.to_string().into()),
                    )
                })?;
                let current = backing.content_generation();
                let next = current.next().map_err(|_| {
                    DataAccessFault::new(
                        address_space,
                        virtual_address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::ContentGenerationExhausted,
                    )
                })?;
                entry.insert((current, next));
            }
            let remaining_in_page = SYNTHETIC_PAGE_SIZE - page_offset(virtual_address);
            cursor = cursor.saturating_add(remaining_in_page.min((end - cursor) as usize) as u64);
        }

        let mut invalidation_kinds = Vec::new();
        invalidation_kinds
            .try_reserve(pending_generations.len())
            .map_err(|_| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::HostBacking(
                        "memory invalidation allocation failed".into(),
                    ),
                )
            })?;
        for physical_slot in pending_generations.keys().copied() {
            if let Some(first) = inner.executable_content_page(physical_slot) {
                invalidation_kinds.push(MemoryInvalidationKind::ExecutableContent {
                    first,
                    second: None,
                });
            }
        }
        let invalidation = (!invalidation_kinds.is_empty())
            .then(|| {
                self.invalidations
                    .reserve_many_from(&invalidation_kinds, MemoryInvalidationOrigin::HostWrite)
            })
            .transpose()
            .map_err(|reason| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::HostBacking(reason.to_string().into()),
                )
            })?;

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
            result.map_err(|reason| {
                DataAccessFault::new(
                    address_space,
                    virtual_address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::HostBacking(reason.to_string().into()),
                )
            })?;
            copied += count;
        }
        if let Some(invalidation) = invalidation {
            invalidation.commit();
        }
        Ok(())
    }

    /// Publishes an alias mapping for an existing physical page.
    pub fn map_page(
        &mut self,
        address_space: AddressSpaceId,
        virtual_address: GuestVirtualAddress,
        physical_page: GuestPhysicalPageId,
        permissions: MemoryPermissions,
    ) -> bool {
        if !virtual_address.is_aligned_to(PAGE_SIZE) {
            return false;
        }
        let invalidations = &self.invalidations;
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        let Some(&physical_slot) = inner.slots_by_id.get(&physical_page) else {
            return false;
        };
        let virtual_page = virtual_page(virtual_address);
        if inner.mappings.get(address_space, virtual_page).is_some() {
            return false;
        }
        let Ok(invalidation) = invalidations.reserve(MemoryInvalidationKind::Mapping {
            address_space,
            start: virtual_address,
            size: PAGE_SIZE,
        }) else {
            return false;
        };
        let Some(mapping_generation) = take_mapping_generation(&mut inner.next_mapping_generation)
        else {
            return false;
        };
        inner.insert_mapping(
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
        invalidation.commit();
        true
    }

    fn fetch<const N: usize>(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<([u8; N], CodeDependencies), InstructionFetchFault> {
        if !address.is_aligned_to(4) {
            return Err(InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::Misaligned,
            ));
        }
        let inner = self.lock_inner();
        let end_offset = page_offset(address) + N;
        // A64 instructions are four-byte aligned and cannot cross this page.
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
        if !backing.observe_executable_content(Arc::clone(&self.invalidations)) {
            return Err(InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::Memory(
                    "canonical page belongs to a different invalidation source".into(),
                ),
            ));
        }
        let mut bytes = [0; N];
        backing
            .read(page_offset(address), &mut bytes)
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
                mapping_generation: mapping.mapping_generation,
            }),
        ))
    }

    fn atomic_transaction(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        operation: impl Fn(MemoryValue) -> (MemoryValue, bool),
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        if access.class != MemoryAccessClass::Atomic || access.alignment != MemoryAlignment::Natural
        {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::InvalidAtomicAccess,
            ));
        }
        let backing = {
            let inner = self.lock_inner();
            resolve_access(&inner, address_space, address, access, DataAccessKind::Read)?;
            let resolved = resolve_access(
                &inner,
                address_space,
                address,
                access,
                DataAccessKind::Write,
            )?;
            if resolved.second.is_some() || resolved.region != MemoryRegionKind::Ram {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::AtomicRegionUnsupported,
                ));
            }
            let ExecutionPhysicalPage::Ram(backing) = inner
                .page(resolved.first.physical_slot)
                .expect("resolved atomic RAM page exists")
            else {
                unreachable!()
            };
            backing.clone()
        };
        backing.prepare_cpu_access().map_err(|reason| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::HostBacking(reason.to_string().into()),
            )
        })?;
        backing.prepare_write().map_err(|reason| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::HostBacking(reason.to_string().into()),
            )
        })?;
        let offset = page_offset(address);
        let byte_count = access.size.bytes();
        loop {
            let mut bytes = [0_u8; 16];
            let generation = backing
                .read_with_generation(offset, &mut bytes[..byte_count])
                .map_err(|reason| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Read,
                        DataAccessFaultReason::HostBacking(reason.to_string().into()),
                    )
                })?;
            let previous = MemoryValue::from_le_slice(access.size, &bytes[..byte_count]);
            let (replacement, stored) = operation(previous);
            if replacement.size() != access.size {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::ValueSizeMismatch,
                ));
            }
            if !stored {
                super::contracts::complete_ordered_read(access.ordering);
                return Ok(AtomicMemoryResult {
                    previous,
                    stored: false,
                    region: MemoryRegionKind::Ram,
                });
            }
            let next_generation = generation.next().map_err(|_| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::ContentGenerationExhausted,
                )
            })?;
            replacement.copy_le_bytes(&mut bytes[..byte_count]);
            super::contracts::begin_ordered_write(access.ordering);
            match backing.write_preflighted(
                offset,
                &bytes[..byte_count],
                generation,
                next_generation,
            ) {
                Ok(()) => {
                    super::contracts::complete_ordered_read(access.ordering);
                    return Ok(AtomicMemoryResult {
                        previous,
                        stored: true,
                        region: MemoryRegionKind::Ram,
                    });
                }
                Err(CanonicalPageError::StaleGeneration { .. }) => continue,
                Err(reason) => {
                    return Err(DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::HostBacking(reason.to_string().into()),
                    ));
                }
            }
        }
    }
}

impl MemoryInvalidationSource for ExecutionMemory {
    fn invalidation_cursor(&self) -> MemoryInvalidationCursor {
        self.invalidations.cursor()
    }

    fn invalidation_signal(&self) -> &AtomicU64 {
        self.invalidations.cursor_signal()
    }

    fn read_invalidations_since(
        &self,
        after: MemoryInvalidationCursor,
        output: &mut Vec<MemoryInvalidation>,
    ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError> {
        self.invalidations.read_since(after, output)
    }
}

impl InstructionMemory for ExecutionMemory {
    fn code_page_span(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<CodePageSpan, InstructionFetchFault> {
        let page_start = page_address(virtual_page(address));
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

    fn fetch32(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault> {
        let (bytes, dependencies) = self.fetch::<4>(address_space, address)?;
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
            let segment = CanonicalBackingSegment::new(
                backing,
                offset,
                count,
                mapping.permissions,
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

fn resolve_access(
    inner: &ExecutionMemoryInner,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    access: MemoryAccess,
    kind: DataAccessKind,
) -> Result<ResolvedDataAccess<ExecutionMapping>, DataAccessFault> {
    resolve_data_access(address_space, address, access, kind, |current| {
        let mapping = inner.mapping_at(address_space, current)?;
        let region = match inner.page(mapping.physical_slot) {
            Some(ExecutionPhysicalPage::Ram(_)) => MemoryRegionKind::Ram,
            Some(ExecutionPhysicalPage::Mmio(_)) => MemoryRegionKind::Device,
            None => return None,
        };
        Some((mapping, mapping.permissions, region))
    })
}

fn bulk_translation_fault(
    error: CanonicalRangeTranslationError,
    kind: DataAccessKind,
) -> DataAccessFault {
    let reason = match error.reason {
        CanonicalRangeTranslationErrorReason::AddressOverflow
        | CanonicalRangeTranslationErrorReason::Empty => DataAccessFaultReason::AddressOverflow,
        CanonicalRangeTranslationErrorReason::Unmapped => DataAccessFaultReason::Unmapped,
        CanonicalRangeTranslationErrorReason::PermissionDenied => match kind {
            DataAccessKind::Read => DataAccessFaultReason::ReadPermissionDenied,
            DataAccessKind::Write => DataAccessFaultReason::WritePermissionDenied,
        },
        CanonicalRangeTranslationErrorReason::DeviceMemory => {
            DataAccessFaultReason::Device("bulk guest-memory transfers require RAM".into())
        }
        CanonicalRangeTranslationErrorReason::InconsistentBacking
        | CanonicalRangeTranslationErrorReason::ResourceExhausted => {
            DataAccessFaultReason::HostBacking(error.to_string().into())
        }
    };
    DataAccessFault::new(error.address_space, error.address, kind, reason)
}

impl CpuMemory for ExecutionMemory {
    fn cpu_memory_backend(&self, address_space: AddressSpaceId) -> CpuMemoryBackend {
        ExecutionMemory::cpu_memory_backend(self, address_space)
            .unwrap_or(CpuMemoryBackend::Checked)
    }

    fn cpu_memory_backend_reason(&self, address_space: AddressSpaceId) -> Box<str> {
        ExecutionMemory::cpu_memory_backend_reason(self, address_space)
            .unwrap_or_else(|| "checked backend used before explicit process binding".into())
    }

    fn direct_address_space_view(
        &self,
        address_space: AddressSpaceId,
    ) -> Option<DirectAddressSpaceView> {
        ExecutionMemory::direct_address_space_view(self, address_space)
    }

    fn execution_gate_identity(&self) -> Option<usize> {
        Some(self.execution_gate.identity())
    }

    fn acquire_execution_lease(&self) -> Option<ExecutionMemoryLease<'_>> {
        Some(ExecutionMemory::acquire_execution_lease(self))
    }

    fn resolve_direct_fault(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        size: crate::memory::MemoryAccessSize,
        kind: DataAccessKind,
    ) -> DirectFaultResolution {
        let fault = |address, reason| {
            DirectFaultResolution::Fault(DataAccessFault::new(address_space, address, kind, reason))
        };
        let byte_count = size.bytes();
        if byte_count > SYNTHETIC_PAGE_SIZE
            || page_offset(address) + byte_count > SYNTHETIC_PAGE_SIZE
            || !address.get().is_multiple_of(byte_count as u64)
        {
            return fault(
                address,
                DataAccessFaultReason::HostBacking(
                    "operation shape is not eligible for a raw direct access".into(),
                ),
            );
        }
        let inner = self.lock_inner();
        let Some(mapping) = inner.mapping_at(address_space, address) else {
            return fault(address, DataAccessFaultReason::Unmapped);
        };
        let required = match kind {
            DataAccessKind::Read => MemoryPermissions::READ,
            DataAccessKind::Write => MemoryPermissions::WRITE,
        };
        if !mapping.permissions.contains(required) {
            return fault(
                address,
                match kind {
                    DataAccessKind::Read => DataAccessFaultReason::ReadPermissionDenied,
                    DataAccessKind::Write => DataAccessFaultReason::WritePermissionDenied,
                },
            );
        }
        let backing = match inner.page(mapping.physical_slot) {
            Some(ExecutionPhysicalPage::Ram(backing)) => backing,
            Some(ExecutionPhysicalPage::Mmio(_)) => {
                return fault(
                    address,
                    DataAccessFaultReason::Device(
                        "device memory requires an explicitly typed access".into(),
                    ),
                );
            }
            None => {
                return fault(
                    address,
                    DataAccessFaultReason::HostBacking(
                        "guest mapping has no physical backing".into(),
                    ),
                );
            }
        };
        let backing = backing.clone();
        // The pre-cutover scalar store frontend owns the complete store value
        // and may finish valid RAM synchronously. Phase 5 removes this outcome
        // when raw JIT stores replace DirectStoreControl.
        if kind == DataAccessKind::Write {
            return DirectFaultResolution::CheckedStore;
        }
        // Fault resolution runs while the vCPU retains its shared execution
        // lease. Never hold the mapping mutex while entering a backing-page
        // transition; mapping/protection revalidation reacquires it afterwards.
        drop(inner);
        let visibility_before = backing.visibility_state();
        if let Err(error) = backing.prepare_cpu_access() {
            return match error {
                CanonicalPageError::Visibility(error) => fault(
                    address,
                    DataAccessFaultReason::HostBacking(error.to_string().into()),
                ),
                error => DirectFaultResolution::Fatal(error.to_string().into_boxed_str()),
            };
        }
        let expected = matches!(
            effective_direct_protection(maximum_direct_protection(mapping), &backing),
            DirectProtection::Read | DirectProtection::ReadWrite
        );
        if expected {
            let protection = self
                .lock_inner()
                .backends
                .get(&address_space)
                .and_then(|binding| match binding {
                    ExecutionBackendBinding::LinuxDirect { arena, .. } => {
                        arena.protection_at(page_address(virtual_page(address)).get())
                    }
                    ExecutionBackendBinding::Checked => None,
                });
            let published = matches!(
                protection,
                Some(DirectProtection::Read | DirectProtection::ReadWrite)
            );
            if !published {
                DirectFaultResolution::Fatal(
                    format!(
                        "eligible direct RAM did not publish the required host protection: address={address:?} protection={protection:?} visibility_before={visibility_before:?}"
                    )
                    .into_boxed_str(),
                )
            } else if matches!(
                visibility_before,
                nixe_memory::VisibilityState::GpuNewer { .. }
            ) {
                DirectFaultResolution::Retry
            } else {
                DirectFaultResolution::Fatal(
                    "eligible CPU-visible direct RAM faulted despite published protection".into(),
                )
            }
        } else {
            fault(
                address,
                DataAccessFaultReason::HostBacking(
                    "guest mapping is not eligible for a raw direct read".into(),
                ),
            )
        }
    }

    fn complete_direct_write_fault(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<DataWriteResult, DataAccessFault> {
        let result = self.write(address_space, address, access, value)?;
        let backing = {
            let inner = self.lock_inner();
            let Some(mapping) = inner.mapping_at(address_space, address) else {
                return Ok(result);
            };
            if !mapping.permissions.contains(MemoryPermissions::WRITE)
                || !matches!(
                    inner.backends.get(&address_space),
                    Some(ExecutionBackendBinding::LinuxDirect { .. })
                )
            {
                return Ok(result);
            }
            match inner.page(mapping.physical_slot) {
                Some(ExecutionPhysicalPage::Ram(backing)) => Some(backing.clone()),
                _ => None,
            }
        };
        if let Some(backing) = backing
            && let Err(error) = backing.resolve_direct_write_fault()
        {
            self.lock_inner().direct_failure = Some(error.to_string().into_boxed_str());
        }
        Ok(result)
    }

    fn read(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault> {
        let mut inner = self.lock_inner();
        let resolved =
            resolve_access(&inner, address_space, address, access, DataAccessKind::Read)?;
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
            super::contracts::complete_ordered_read(access.ordering);
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
        super::contracts::complete_ordered_read(access.ordering);
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
        let resolved = resolve_access(
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
            super::contracts::begin_ordered_write(access.ordering);
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
        super::contracts::begin_ordered_write(access.ordering);
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
                let offset = page_offset(address);
                loop {
                    let generation = backing.content_generation();
                    let next_generation = generation.next().map_err(|_| {
                        DataAccessFault::new(
                            address_space,
                            address,
                            DataAccessKind::Write,
                            DataAccessFaultReason::ContentGenerationExhausted,
                        )
                    })?;
                    match backing.write_preflighted(
                        offset,
                        &bytes[..byte_count],
                        generation,
                        next_generation,
                    ) {
                        Ok(()) => break,
                        Err(CanonicalPageError::StaleGeneration { .. }) => continue,
                        Err(reason) => {
                            return Err(DataAccessFault::new(
                                address_space,
                                address,
                                DataAccessKind::Write,
                                DataAccessFaultReason::HostBacking(reason.to_string().into()),
                            ));
                        }
                    }
                }
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

    fn atomic_read_modify_write(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        kind: AtomicRmwKind,
        operand: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        if operand.size() != access.size {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ValueSizeMismatch,
            ));
        }
        self.atomic_transaction(address_space, address, access, |previous| {
            (
                kind.apply(previous, operand)
                    .expect("validated atomic operands have one width"),
                true,
            )
        })
    }

    fn atomic_compare_exchange(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        expected: MemoryValue,
        replacement: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        if expected.size() != access.size || replacement.size() != access.size {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ValueSizeMismatch,
            ));
        }
        self.atomic_transaction(address_space, address, access, |previous| {
            if previous == expected {
                (replacement, true)
            } else {
                (previous, false)
            }
        })
    }

    fn maintain_cache(
        &self,
        address_space: AddressSpaceId,
        kind: super::CacheMaintenanceKind,
        address: Option<GuestVirtualAddress>,
    ) -> Result<(), DataAccessFault> {
        if kind == super::CacheMaintenanceKind::InstructionInvalidate && address.is_none() {
            self.invalidations
                .reserve(MemoryInvalidationKind::InstructionCache { address_space })
                .map_err(|reason| {
                    DataAccessFault::new(
                        address_space,
                        GuestVirtualAddress::MIN,
                        DataAccessKind::Read,
                        DataAccessFaultReason::HostBacking(reason.to_string().into()),
                    )
                })?
                .commit();
            return Ok(());
        }
        let address = address.ok_or_else(|| {
            DataAccessFault::new(
                address_space,
                GuestVirtualAddress::MIN,
                DataAccessKind::Read,
                DataAccessFaultReason::AddressOverflow,
            )
        })?;
        let inner = self.lock_inner();
        let mapping = inner.mapping_at(address_space, address).ok_or_else(|| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::Unmapped,
            )
        })?;
        let Some(ExecutionPhysicalPage::Ram(backing)) = inner.page(mapping.physical_slot) else {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::Device("cache maintenance requires canonical RAM".into()),
            ));
        };
        match kind {
            super::CacheMaintenanceKind::InstructionInvalidate => {
                self.invalidations
                    .reserve_with_origin(
                        MemoryInvalidationKind::ExecutableContent {
                            first: mapping.physical_page,
                            second: None,
                        },
                        MemoryInvalidationOrigin::CacheMaintenance,
                    )
                    .map_err(|reason| {
                        DataAccessFault::new(
                            address_space,
                            address,
                            DataAccessKind::Read,
                            DataAccessFaultReason::HostBacking(reason.to_string().into()),
                        )
                    })?
                    .commit();
            }
            super::CacheMaintenanceKind::DataInvalidate
            | super::CacheMaintenanceKind::DataClean
            | super::CacheMaintenanceKind::DataCleanAndInvalidate => {
                let generation = backing.content_generation();
                let reservation = inner
                    .executable_content_page(mapping.physical_slot)
                    .map(|first| {
                        self.invalidations.reserve_with_origin(
                            MemoryInvalidationKind::ExecutableContent {
                                first,
                                second: None,
                            },
                            MemoryInvalidationOrigin::CacheMaintenance,
                        )
                    })
                    .transpose()
                    .map_err(|reason| {
                        DataAccessFault::new(
                            address_space,
                            address,
                            DataAccessKind::Read,
                            DataAccessFaultReason::HostBacking(reason.to_string().into()),
                        )
                    })?;
                backing.prepare_cpu_access().map_err(|reason| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Read,
                        DataAccessFaultReason::HostBacking(reason.to_string().into()),
                    )
                })?;
                if backing.content_generation() != generation
                    && let Some(reservation) = reservation
                {
                    reservation.commit();
                }
            }
            super::CacheMaintenanceKind::InstructionPrefetch => {}
        }
        Ok(())
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
        let page = virtual_page(address);
        let end_page = virtual_page(end_exclusive);
        let state = inner.mapping_state(address_space, page);

        let (first_page, last_page_exclusive) = if let Some(state) = state {
            coalesce_mapped_pages(page, end_page, state, |page| {
                inner.mapping_state(address_space, page)
            })
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
        memory_query_result(first_page, last_page_exclusive, state)
    }

    fn load_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<(DataReadResult, ExclusiveReservation), DataAccessFault> {
        let inner = self.lock_inner();
        let resolved =
            resolve_access(&inner, address_space, address, access, DataAccessKind::Read)?;
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
        let _generation = backing
            .read_with_generation(page_offset(address), &mut bytes[..byte_count])
            .map_err(|reason| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Read,
                    DataAccessFaultReason::HostBacking(reason.to_string().into()),
                )
            })?;
        super::contracts::complete_ordered_read(access.ordering);
        let value = MemoryValue::from_le_slice(access.size, &bytes[..byte_count]);
        Ok((
            DataReadResult {
                value,
                region: MemoryRegionKind::Ram,
            },
            ExclusiveReservation {
                page: mapping.physical_page,
                byte_offset: page_offset(address) as u16,
                access_size: access.size.bytes() as u8,
                expected: value,
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
        let resolved = resolve_access(
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
        let Some(ExecutionPhysicalPage::Ram(_)) = inner.page(resolved.first.physical_slot) else {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::MixedRegions,
            ));
        };
        let matches = reservation.page == resolved.first.physical_page
            && usize::from(reservation.byte_offset) == page_offset(address)
            && usize::from(reservation.access_size) == access.size.bytes();
        drop(inner);
        if !matches {
            return Ok((
                DataWriteResult {
                    region: MemoryRegionKind::Ram,
                },
                false,
            ));
        }
        let atomic_access = MemoryAccess::new(
            access.size,
            MemoryAlignment::Natural,
            access.ordering,
            MemoryAccessClass::Atomic,
        );
        let result = self.atomic_compare_exchange(
            address_space,
            address,
            atomic_access,
            reservation.expected,
            value,
        )?;
        Ok((
            DataWriteResult {
                region: MemoryRegionKind::Ram,
            },
            result.stored,
        ))
    }
}

impl ProcessMemory for ExecutionMemory {
    fn read_bytes(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        output: &mut [u8],
    ) -> Result<(), DataAccessFault> {
        if output.is_empty() {
            return Ok(());
        }
        let size = u64::try_from(output.len()).map_err(|_| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::AddressOverflow,
            )
        })?;
        let range = self
            .translate_canonical_range(address_space, address, size, MemoryPermissions::READ)
            .map_err(|error| bulk_translation_fault(error, DataAccessKind::Read))?;
        range.read(0, output).map_err(|error| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::HostBacking(error.to_string().into()),
            )
        })
    }

    fn write_bytes(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        bytes: &[u8],
    ) -> Result<(), DataAccessFault> {
        if bytes.is_empty() {
            return Ok(());
        }
        let size = u64::try_from(bytes.len()).map_err(|_| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::AddressOverflow,
            )
        })?;
        let range = self
            .translate_canonical_range(address_space, address, size, MemoryPermissions::WRITE)
            .map_err(|error| bulk_translation_fault(error, DataAccessKind::Write))?;
        let invalidation_kinds = {
            let inner = self.lock_inner();
            let end = address
                .get()
                .checked_add(size)
                .expect("bulk range was checked");
            let mut kinds = Vec::new();
            let page_count = page_offset(address)
                .checked_add(bytes.len())
                .map(|span| span.div_ceil(SYNTHETIC_PAGE_SIZE))
                .ok_or_else(|| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::AddressOverflow,
                    )
                })?;
            kinds.try_reserve(page_count).map_err(|_| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::HostBacking(
                        "memory invalidation allocation failed".into(),
                    ),
                )
            })?;
            let mut cursor = address.get();
            while cursor < end {
                let current = GuestVirtualAddress::new(cursor);
                let mapping = inner
                    .mapping_at(address_space, current)
                    .expect("canonical range translation validated every mapping");
                if let Some(first) = inner.executable_content_page(mapping.physical_slot)
                    && !kinds.iter().any(|kind| {
                        matches!(kind, MemoryInvalidationKind::ExecutableContent { first: page, .. } if *page == first)
                    })
                {
                    kinds.push(MemoryInvalidationKind::ExecutableContent {
                        first,
                        second: None,
                    });
                }
                cursor += (SYNTHETIC_PAGE_SIZE - page_offset(current)).min((end - cursor) as usize)
                    as u64;
            }
            kinds
        };
        let invalidation = (!invalidation_kinds.is_empty())
            .then(|| {
                self.invalidations
                    .reserve_many_from(&invalidation_kinds, MemoryInvalidationOrigin::HostWrite)
            })
            .transpose()
            .map_err(|reason| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::HostBacking(reason.to_string().into()),
                )
            })?;
        let mut batch = CanonicalWriteBatch::new();
        batch.stage(&range, 0, bytes).map_err(|error| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::HostBacking(error.to_string().into()),
            )
        })?;
        batch.commit().map_err(|error| {
            let reason = match error {
                nixe_memory::CanonicalWriteBatchError::GenerationExhausted(_) => {
                    DataAccessFaultReason::ContentGenerationExhausted
                }
                error => DataAccessFaultReason::HostBacking(error.to_string().into()),
            };
            DataAccessFault::new(address_space, address, DataAccessKind::Write, reason)
        })?;
        if let Some(invalidation) = invalidation {
            invalidation.commit();
        }
        Ok(())
    }

    fn resize_zeroed_mapping(
        &self,
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        old_size: u64,
        new_size: u64,
        permissions: MemoryPermissions,
        purpose: MemoryMappingPurpose,
    ) -> Result<(), MemoryMappingError> {
        let error = |address, reason| MemoryMappingError {
            address_space,
            address,
            reason,
        };
        let (Some(old_range), Some(new_range)) = (
            PageRange::new(start, old_size),
            PageRange::new(start, new_size),
        ) else {
            return Err(error(start, MemoryMappingErrorReason::InvalidRange));
        };
        if writable_executable(permissions) {
            return Err(error(start, MemoryMappingErrorReason::WritableExecutable));
        }

        let first_page = old_range.first;
        let old_end_page = old_range.end;
        let new_end_page = new_range.end;
        let backing_store = self.backing_store.clone();
        let mut mutation = self.begin_mapping_mutation();
        let mut inner = self.lock_inner();
        for page in first_page..old_end_page {
            let Some(mapping) = inner.mappings.get(address_space, page) else {
                return Err(error(
                    page_address(page),
                    MemoryMappingErrorReason::MappingStateMismatch,
                ));
            };
            if mapping.purpose != purpose || mapping.permissions != permissions {
                return Err(error(
                    page_address(page),
                    MemoryMappingErrorReason::MappingStateMismatch,
                ));
            }
        }
        for page in old_end_page..new_end_page {
            if inner.mappings.get(address_space, page).is_some() {
                return Err(error(
                    page_address(page),
                    MemoryMappingErrorReason::AlreadyMapped,
                ));
            }
        }

        if new_end_page < old_end_page {
            let invalidation = self
                .invalidations
                .reserve(MemoryInvalidationKind::Mapping {
                    address_space,
                    start,
                    size: old_size.max(new_size),
                })
                .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
            let removed_pages = usize::try_from(old_end_page - new_end_page)
                .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
            inner
                .free_physical_slots
                .try_reserve(removed_pages)
                .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
            inner.revoke_direct_mapping_range(address_space, new_end_page, old_end_page);
            for page in new_end_page..old_end_page {
                let mapping = inner
                    .remove_mapping_unpublished(address_space, page)
                    .expect("shrinking range was preflighted");
                if inner.mapping_count(mapping.physical_slot) == 0 {
                    inner.remove_page(mapping.physical_page, mapping.physical_slot);
                }
            }
            mutation.commit();
            invalidation.commit();
            return Ok(());
        }
        if new_end_page == old_end_page {
            return Ok(());
        }

        let invalidation = self
            .invalidations
            .reserve(MemoryInvalidationKind::Mapping {
                address_space,
                start,
                size: old_size.max(new_size),
            })
            .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
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
            let physical_page = allocate_page_id(&mut next_page_id, |page| {
                inner.slots_by_id.contains_key(&page)
            })
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
            take_mapping_generation(&mut inner.next_mapping_generation)
                .ok_or_else(|| error(start, MemoryMappingErrorReason::GenerationExhausted))?
        };
        for (page, physical_page, backing) in pending {
            let slot = inner
                .push_page(physical_page, ExecutionPhysicalPage::Ram(backing))
                .expect("allocated physical identity is unique");
            inner.insert_mapping_unpublished(
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
        }
        inner.publish_direct_mapping_range(address_space, old_end_page, new_end_page);
        inner.next_page_id = next_page_id;
        mutation.commit();
        invalidation.commit();
        Ok(())
    }

    fn map_alias(&self, request: MemoryAliasRequest) -> Result<(), MemoryAliasError> {
        let MemoryAliasRequest {
            address_space,
            destination,
            source,
            size,
            source_before,
            source_after,
            destination_properties,
        } = request;
        let error = |address, reason| MemoryAliasError {
            address_space,
            address,
            reason,
        };
        let (Some(source_range), Some(destination_range)) = (
            PageRange::new(source, size),
            PageRange::new(destination, size),
        ) else {
            return Err(error(source, MemoryAliasErrorReason::InvalidRange));
        };
        if source_range.is_empty() {
            return Err(error(source, MemoryAliasErrorReason::InvalidRange));
        }

        let mut mutation = self.begin_mapping_mutation();
        let mut inner = self.lock_inner();
        for (source_page, destination_page) in (source_range.first..source_range.end)
            .zip(destination_range.first..destination_range.end)
        {
            let Some(source_mapping) = inner.mappings.get(address_space, source_page) else {
                return Err(error(
                    page_address(source_page),
                    MemoryAliasErrorReason::SourceStateMismatch,
                ));
            };
            if source_mapping.properties() != source_before
                || !matches!(
                    inner.page(source_mapping.physical_slot),
                    Some(ExecutionPhysicalPage::Ram(_))
                )
            {
                return Err(error(
                    page_address(source_page),
                    MemoryAliasErrorReason::SourceStateMismatch,
                ));
            }
            if inner
                .mappings
                .get(address_space, destination_page)
                .is_some()
            {
                return Err(error(
                    page_address(destination_page),
                    MemoryAliasErrorReason::DestinationStateMismatch,
                ));
            }
        }

        let invalidation_kinds = [
            MemoryInvalidationKind::Mapping {
                address_space,
                start: source,
                size,
            },
            MemoryInvalidationKind::Mapping {
                address_space,
                start: destination,
                size,
            },
        ];
        let invalidation = self
            .invalidations
            .reserve_many(&invalidation_kinds)
            .map_err(|_| error(source, MemoryAliasErrorReason::ResourceExhausted))?;
        let mapping_generation = take_mapping_generation(&mut inner.next_mapping_generation)
            .ok_or_else(|| error(source, MemoryAliasErrorReason::GenerationExhausted))?;

        for (source_page, destination_page) in (source_range.first..source_range.end)
            .zip(destination_range.first..destination_range.end)
        {
            let source_mapping = inner
                .mappings
                .get(address_space, source_page)
                .expect("alias source range was preflighted");
            inner.set_mapping_properties(
                address_space,
                source_page,
                source_after,
                mapping_generation,
            );
            inner.insert_mapping_unpublished(
                address_space,
                destination_page,
                ExecutionMapping {
                    physical_page: source_mapping.physical_page,
                    physical_slot: source_mapping.physical_slot,
                    mapping_generation,
                    permissions: destination_properties.permissions,
                    purpose: destination_properties.purpose,
                    attributes: destination_properties.attributes,
                },
            );
        }
        inner.publish_direct_mapping_range(address_space, source_range.first, source_range.end);
        inner.publish_direct_mapping_range(
            address_space,
            destination_range.first,
            destination_range.end,
        );
        mutation.commit();
        invalidation.commit();
        Ok(())
    }

    fn unmap_alias(&self, request: MemoryAliasRequest) -> Result<(), MemoryAliasError> {
        let MemoryAliasRequest {
            address_space,
            destination,
            source,
            size,
            source_before,
            source_after,
            destination_properties,
        } = request;
        let error = |address, reason| MemoryAliasError {
            address_space,
            address,
            reason,
        };
        let (Some(source_range), Some(destination_range)) = (
            PageRange::new(source, size),
            PageRange::new(destination, size),
        ) else {
            return Err(error(source, MemoryAliasErrorReason::InvalidRange));
        };
        if source_range.is_empty() {
            return Err(error(source, MemoryAliasErrorReason::InvalidRange));
        }

        let mut mutation = self.begin_mapping_mutation();
        let mut inner = self.lock_inner();
        for (source_page, destination_page) in (source_range.first..source_range.end)
            .zip(destination_range.first..destination_range.end)
        {
            let Some(source_mapping) = inner.mappings.get(address_space, source_page) else {
                return Err(error(
                    page_address(source_page),
                    MemoryAliasErrorReason::SourceStateMismatch,
                ));
            };
            if source_mapping.properties() != source_before {
                return Err(error(
                    page_address(source_page),
                    MemoryAliasErrorReason::SourceStateMismatch,
                ));
            }
            let Some(destination_mapping) = inner.mappings.get(address_space, destination_page)
            else {
                return Err(error(
                    page_address(destination_page),
                    MemoryAliasErrorReason::DestinationStateMismatch,
                ));
            };
            if destination_mapping.properties() != destination_properties {
                return Err(error(
                    page_address(destination_page),
                    MemoryAliasErrorReason::DestinationStateMismatch,
                ));
            }
            if source_mapping.physical_page != destination_mapping.physical_page
                || source_mapping.physical_slot != destination_mapping.physical_slot
            {
                return Err(error(
                    page_address(destination_page),
                    MemoryAliasErrorReason::PhysicalIdentityMismatch,
                ));
            }
        }

        let invalidation_kinds = [
            MemoryInvalidationKind::Mapping {
                address_space,
                start: source,
                size,
            },
            MemoryInvalidationKind::Mapping {
                address_space,
                start: destination,
                size,
            },
        ];
        let invalidation = self
            .invalidations
            .reserve_many(&invalidation_kinds)
            .map_err(|_| error(source, MemoryAliasErrorReason::ResourceExhausted))?;
        let mapping_generation = take_mapping_generation(&mut inner.next_mapping_generation)
            .ok_or_else(|| error(source, MemoryAliasErrorReason::GenerationExhausted))?;

        inner.revoke_direct_mapping_range(
            address_space,
            destination_range.first,
            destination_range.end,
        );
        for (source_page, destination_page) in (source_range.first..source_range.end)
            .zip(destination_range.first..destination_range.end)
        {
            let removed = inner
                .remove_mapping_unpublished(address_space, destination_page)
                .expect("alias destination range was preflighted");
            debug_assert_ne!(inner.mapping_count(removed.physical_slot), 0);
            inner.set_mapping_properties(
                address_space,
                source_page,
                source_after,
                mapping_generation,
            );
        }
        inner.publish_direct_mapping_range(address_space, source_range.first, source_range.end);
        mutation.commit();
        invalidation.commit();
        Ok(())
    }

    fn set_permissions(
        &self,
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        size: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryProtectionError> {
        let error = |address, reason| MemoryProtectionError {
            address_space,
            address,
            reason,
        };
        let Some(range) = PageRange::new(start, size).filter(|range| !range.is_empty()) else {
            return Err(error(start, MemoryProtectionErrorReason::InvalidRange));
        };
        if writable_executable(permissions) {
            return Err(error(
                start,
                MemoryProtectionErrorReason::WritableExecutable,
            ));
        }
        let mut mutation = self.begin_mapping_mutation();
        let mut inner = self.lock_inner();
        for page in range.first..range.end {
            let Some(mapping) = inner.mappings.get(address_space, page) else {
                return Err(error(
                    page_address(page),
                    MemoryProtectionErrorReason::Unmapped,
                ));
            };
            if mapping
                .attributes
                .contains(MemoryAttributes::PERMISSION_LOCKED)
                && mapping.permissions != permissions
            {
                return Err(error(
                    page_address(page),
                    MemoryProtectionErrorReason::PermissionLocked,
                ));
            }
        }
        let changed = (range.first..range.end).any(|page| {
            inner
                .mappings
                .get(address_space, page)
                .is_some_and(|mapping| mapping.permissions != permissions)
        });
        if !changed {
            return Ok(());
        }
        let invalidation = self
            .invalidations
            .reserve(MemoryInvalidationKind::Mapping {
                address_space,
                start,
                size,
            })
            .map_err(|_| error(start, MemoryProtectionErrorReason::GenerationExhausted))?;
        let mapping_generation = take_mapping_generation(&mut inner.next_mapping_generation)
            .ok_or_else(|| error(start, MemoryProtectionErrorReason::GenerationExhausted))?;
        for page in range.first..range.end {
            inner.set_mapping_permissions(address_space, page, permissions, mapping_generation);
        }
        inner.publish_direct_mapping_range(address_space, range.first, range.end);
        mutation.commit();
        invalidation.commit();
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
        let error = |address, reason| MemoryProtectionError {
            address_space,
            address,
            reason,
        };
        let Some(range) = PageRange::new(start, size).filter(|range| !range.is_empty()) else {
            return Err(error(start, MemoryProtectionErrorReason::InvalidRange));
        };
        if masked_attributes(MemoryAttributes::NONE, mask, value).is_none() {
            return Err(error(start, MemoryProtectionErrorReason::InvalidRange));
        }
        let mut mutation = self.begin_mapping_mutation();
        let mut inner = self.lock_inner();
        for page in range.first..range.end {
            if inner.mappings.get(address_space, page).is_none() {
                return Err(error(
                    page_address(page),
                    MemoryProtectionErrorReason::Unmapped,
                ));
            }
        }
        let changed = (range.first..range.end).any(|page| {
            let mapping = inner
                .mappings
                .get(address_space, page)
                .expect("attribute range was preflighted");
            mapping.attributes != masked_attributes(mapping.attributes, mask, value).unwrap()
        });
        if !changed {
            return Ok(());
        }
        let invalidation = self
            .invalidations
            .reserve(MemoryInvalidationKind::Mapping {
                address_space,
                start,
                size,
            })
            .map_err(|_| error(start, MemoryProtectionErrorReason::GenerationExhausted))?;
        let mapping_generation = take_mapping_generation(&mut inner.next_mapping_generation)
            .ok_or_else(|| error(start, MemoryProtectionErrorReason::GenerationExhausted))?;
        for page in range.first..range.end {
            let mapping = inner
                .mappings
                .get_mut(address_space, page)
                .expect("attribute range was preflighted");
            mapping.attributes = masked_attributes(mapping.attributes, mask, value)
                .expect("attribute mask was validated");
            mapping.mapping_generation = mapping_generation;
        }
        inner.publish_direct_mapping_range(address_space, range.first, range.end);
        mutation.commit();
        invalidation.commit();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::memory::{CacheMaintenanceKind, MemoryAccessSize};
    use nixe_memory::{
        CpuVisibilityRequest, DIRECT_PAGE_SIZE, DeviceAccessDeclaration, DeviceVisibilityPoint,
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
    fn cpu_memory_backend_selection_is_immutable_after_binding() {
        let mut memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(7);
        assert_eq!(
            memory
                .bind_cpu_memory_backend(space, 0x4000, DirectBackendPolicy::Disabled)
                .unwrap(),
            CpuMemoryBackend::Checked
        );
        assert!(
            memory
                .bind_cpu_memory_backend(space, 0x4000, DirectBackendPolicy::Required)
                .unwrap_err()
                .to_string()
                .contains("already bound")
        );
        assert_eq!(
            memory.cpu_memory_backend(space),
            Some(CpuMemoryBackend::Checked)
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
        assert_eq!(
            memory
                .write_bytes(space, GuestVirtualAddress::new(0x1fff), &[0xaa, 0xbb])
                .unwrap_err()
                .reason,
            DataAccessFaultReason::WritePermissionDenied
        );
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
        let mut observed = [0; 1];
        retained.read(7, &mut observed).unwrap();
        assert_eq!(observed, [0x5a]);
    }

    #[test]
    fn cpu_read_reconciles_gpu_newer_backing_through_neutral_slow_path() {
        let memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(10);
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
            .publish_device_write(declaration, Arc::clone(&coordinator))
            .unwrap();
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
            .publish_device_write(second_write, coordinator)
            .unwrap();
        memory
            .write(
                space,
                GuestVirtualAddress::new(0x8009),
                MemoryAccess::normal(crate::memory::MemoryAccessSize::Byte),
                MemoryValue::U8(0x33),
            )
            .unwrap();
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
    fn gpu_write_to_fetched_code_publishes_physical_invalidation_before_writeback() {
        let memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(12);
        let address = GuestVirtualAddress::new(0xa000);
        memory
            .resize_zeroed_mapping(
                space,
                address,
                0,
                0x1000,
                MemoryPermissions::READ_EXECUTE,
                MemoryMappingPurpose::CodeStatic,
            )
            .unwrap();
        memory.fetch32(space, address).unwrap();
        let after_mapping = memory.invalidation_cursor();
        let retained = memory
            .translate_canonical_range(space, address, 0x1000, MemoryPermissions::READ)
            .unwrap();
        let physical_page = retained.segments()[0].page().page();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(DeviceWriteback {
            bytes: vec![0x5a; 0x1000].into_boxed_slice(),
        });
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(6),
            DeviceVisibilityPoint::new(30),
            DeviceVisibilityPoint::new(31),
        )
        .unwrap();
        retained
            .prepare_device_access(declaration, Arc::clone(&coordinator))
            .unwrap();
        retained
            .publish_device_write(declaration, coordinator)
            .unwrap();

        let mut records = Vec::new();
        memory
            .read_invalidations_since(after_mapping, &mut records)
            .unwrap();
        assert_eq!(
            records.as_slice(),
            &[MemoryInvalidation {
                cursor: MemoryInvalidationCursor::new(after_mapping.get() + 1),
                kind: MemoryInvalidationKind::ExecutableContent {
                    first: physical_page,
                    second: None,
                },
                origin: MemoryInvalidationOrigin::DeviceWrite,
            }]
        );
        assert!(matches!(
            retained.segments()[0].visibility_state(),
            VisibilityState::GpuNewer { .. }
        ));
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
            .publish_device_write(declaration, coordinator)
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
    fn direct_backend_represents_every_permission_shape_conservatively() {
        let mut memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(41);
        let cases = [
            (0x1000, MemoryPermissions::READ, DirectProtection::Read),
            (
                0x2000,
                MemoryPermissions::READ_WRITE,
                DirectProtection::Read,
            ),
            (0x3000, MemoryPermissions::WRITE, DirectProtection::None),
            (
                0x4000,
                MemoryPermissions::READ_EXECUTE,
                DirectProtection::Read,
            ),
            (0x5000, MemoryPermissions::EXECUTE, DirectProtection::None),
        ];
        for (index, (address, permissions, _)) in cases.iter().copied().enumerate() {
            let page = GuestPhysicalPageId::new(index as u64 + 1);
            assert!(memory.add_ram_page(page));
            assert!(memory.map_page(space, GuestVirtualAddress::new(address), page, permissions,));
        }
        memory
            .bind_cpu_memory_backend(space, 0x8000, DirectBackendPolicy::Required)
            .unwrap();
        for (address, _, expected) in cases {
            assert_eq!(
                memory.direct_protection_at(space, GuestVirtualAddress::new(address)),
                Some(expected),
            );
        }
    }

    #[test]
    fn physical_slots_track_executable_aliases_across_mapping_transitions() {
        let mut memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(1);
        let physical_page = GuestPhysicalPageId::new(1);
        let writable = GuestVirtualAddress::new(0x1000);
        let executable_alias = GuestVirtualAddress::new(0x2000);
        let access = MemoryAccess::normal(MemoryAccessSize::Byte);

        assert!(memory.add_ram_page(physical_page));
        assert!(memory.map_page(
            space,
            writable,
            physical_page,
            MemoryPermissions::READ_WRITE,
        ));
        memory
            .bind_cpu_memory_backend(space, 0x4000, DirectBackendPolicy::Required)
            .unwrap();
        memory
            .complete_direct_write_fault(space, writable, access, MemoryValue::U8(0x10))
            .unwrap();
        let view = memory.direct_address_space_view(space).unwrap();
        let writable_control = unsafe {
            &*((view.store_controls as *const AtomicUsize)
                .add(writable.get() as usize / DIRECT_PAGE_SIZE))
        };
        assert_ne!(writable_control.load(Ordering::Acquire), 0);
        let arena = match &memory.inner_mut().backends[&space] {
            ExecutionBackendBinding::LinuxDirect { arena, .. } => arena.clone(),
            ExecutionBackendBinding::Checked => panic!("required direct backend was not bound"),
        };
        assert_eq!(
            arena.protection_at(writable.get()),
            Some(DirectProtection::ReadWrite)
        );
        assert!(memory.map_page(
            space,
            executable_alias,
            physical_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        assert_ne!(writable_control.load(Ordering::Acquire), 0);
        assert_eq!(
            arena.protection_at(writable.get()),
            Some(DirectProtection::ReadWrite)
        );

        let physical_slot = memory.inner_mut().slots_by_id[&physical_page];
        {
            let inner = memory.inner_mut();
            let slot = inner.physical_slots[physical_slot].as_ref().unwrap();
            assert_eq!(slot.mapping_count, 2);
            assert_eq!(slot.executable_content_mapping_count, 1);
        }
        let before_write = memory.invalidation_cursor();
        memory
            .write(space, writable, access, MemoryValue::U8(0x11))
            .unwrap();
        let mut invalidations = Vec::new();
        let after_write = memory
            .read_invalidations_since(before_write, &mut invalidations)
            .unwrap();
        assert_eq!(after_write, before_write);
        assert!(invalidations.is_empty());

        memory
            .maintain_cache(
                space,
                CacheMaintenanceKind::InstructionInvalidate,
                Some(executable_alias),
            )
            .unwrap();
        memory
            .read_invalidations_since(after_write, &mut invalidations)
            .unwrap();
        assert!(matches!(
            invalidations.as_slice(),
            [MemoryInvalidation {
                kind: MemoryInvalidationKind::ExecutableContent {
                    first,
                    second: None,
                },
                ..
            }] if *first == physical_page
        ));

        memory
            .set_permissions(space, executable_alias, PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        assert_eq!(
            memory.inner_mut().physical_slots[physical_slot]
                .as_ref()
                .unwrap()
                .executable_content_mapping_count,
            0
        );
        let before_plain_write = memory.invalidation_cursor();
        memory
            .write(space, writable, access, MemoryValue::U8(0x22))
            .unwrap();
        invalidations.clear();
        memory
            .read_invalidations_since(before_plain_write, &mut invalidations)
            .unwrap();
        assert!(invalidations.is_empty());

        assert!(memory.set_mapping_purpose(
            space,
            executable_alias,
            PAGE_SIZE,
            MemoryMappingPurpose::CodeStatic,
        ));
        assert_eq!(
            memory.inner_mut().physical_slots[physical_slot]
                .as_ref()
                .unwrap()
                .executable_content_mapping_count,
            1
        );
        assert!(memory.set_mapping_purpose(
            space,
            executable_alias,
            PAGE_SIZE,
            MemoryMappingPurpose::Normal,
        ));

        memory
            .resize_zeroed_mapping(
                space,
                writable,
                PAGE_SIZE,
                0,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        {
            let inner = memory.lock_inner();
            let slot = inner.physical_slots[physical_slot].as_ref().unwrap();
            assert_eq!(slot.mapping_count, 1);
            assert_eq!(slot.executable_content_mapping_count, 0);
        }
        memory
            .resize_zeroed_mapping(
                space,
                executable_alias,
                PAGE_SIZE,
                0,
                MemoryPermissions::READ,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        assert!(!memory.inner_mut().slots_by_id.contains_key(&physical_page));
        assert!(memory.inner_mut().physical_slots[physical_slot].is_none());
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
    fn direct_backend_eagerly_maps_aliases_and_revokes_them_for_device_ownership() {
        let mut memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(31);
        let first = GuestVirtualAddress::new(0x1000);
        let second = GuestVirtualAddress::new(0x2000);
        let physical = GuestPhysicalPageId::new(7);
        assert!(memory.add_ram_page(physical));
        assert!(memory.map_page(space, first, physical, MemoryPermissions::READ_WRITE));
        assert!(memory.map_page(space, second, physical, MemoryPermissions::READ));
        assert_eq!(
            memory
                .bind_cpu_memory_backend(space, 0x4000, DirectBackendPolicy::Required)
                .unwrap(),
            CpuMemoryBackend::LinuxDirect
        );
        assert_eq!(
            memory.direct_protection_at(space, first),
            Some(DirectProtection::Read)
        );
        assert_eq!(
            memory.direct_protection_at(space, second),
            Some(DirectProtection::Read)
        );

        memory
            .write(
                space,
                first,
                MemoryAccess::normal(crate::memory::MemoryAccessSize::Byte),
                MemoryValue::U8(0x44),
            )
            .unwrap();
        let view = memory.direct_address_space_view(space).unwrap();
        assert_eq!(
            unsafe { ((view.base + second.get() as usize) as *const u8).read() },
            0x44
        );

        let retained = memory
            .translate_canonical_range(space, first, PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(DeviceWriteback {
            bytes: vec![0x77; SYNTHETIC_PAGE_SIZE].into_boxed_slice(),
        });
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(9),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        retained
            .prepare_device_access(declaration, Arc::clone(&coordinator))
            .unwrap();
        retained
            .publish_device_write(declaration, coordinator)
            .unwrap();
        assert_eq!(
            memory.direct_protection_at(space, first),
            Some(DirectProtection::None)
        );
        assert_eq!(
            memory.direct_protection_at(space, second),
            Some(DirectProtection::None)
        );

        let lease = memory.acquire_execution_lease();
        assert_eq!(
            memory
                .read(
                    space,
                    second,
                    MemoryAccess::normal(crate::memory::MemoryAccessSize::Byte),
                )
                .unwrap()
                .value,
            MemoryValue::U8(0x77)
        );
        assert_eq!(
            memory.direct_protection_at(space, first),
            Some(DirectProtection::Read)
        );
        drop(lease);
    }

    #[test]
    fn direct_backend_grows_and_shrinks_zeroed_mappings() {
        let mut memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(32);
        let start = GuestVirtualAddress::new(0x1000);
        let page_count = 32_u64;
        let size = page_count * SYNTHETIC_PAGE_SIZE as u64;
        assert_eq!(
            memory
                .bind_cpu_memory_backend(space, size + 0x2000, DirectBackendPolicy::Required,)
                .unwrap(),
            CpuMemoryBackend::LinuxDirect
        );

        memory
            .resize_zeroed_mapping(
                space,
                start,
                0,
                size,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();
        assert_eq!(
            memory.direct_protection_at(space, start),
            Some(DirectProtection::Read)
        );

        memory
            .resize_zeroed_mapping(
                space,
                start,
                size,
                SYNTHETIC_PAGE_SIZE as u64,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();
        assert_eq!(
            memory.direct_protection_at(space, start),
            Some(DirectProtection::Read)
        );
        assert_eq!(
            memory.direct_protection_at(space, start.wrapping_offset(SYNTHETIC_PAGE_SIZE as i64)),
            None
        );
    }

    #[test]
    fn direct_backend_applies_contiguous_protection_mutations() {
        let mut memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(33);
        let start = GuestVirtualAddress::new(0x10_0000);
        let page_count = 64_u64;
        let size = page_count * SYNTHETIC_PAGE_SIZE as u64;
        memory
            .bind_cpu_memory_backend(
                space,
                start.get() + size + SYNTHETIC_PAGE_SIZE as u64,
                DirectBackendPolicy::Required,
            )
            .unwrap();
        memory
            .resize_zeroed_mapping(
                space,
                start,
                0,
                size,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();

        memory
            .set_attributes(
                space,
                start,
                size,
                MemoryAttributes::UNCACHED,
                MemoryAttributes::UNCACHED,
            )
            .unwrap();
        assert_eq!(
            memory.direct_protection_at(space, start),
            Some(DirectProtection::Read)
        );

        memory
            .set_attributes(
                space,
                start,
                size,
                MemoryAttributes::UNCACHED,
                MemoryAttributes::NONE,
            )
            .unwrap();
        assert_eq!(
            memory.direct_protection_at(space, start),
            Some(DirectProtection::Read)
        );

        memory
            .set_permissions(space, start, size, MemoryPermissions::WRITE)
            .unwrap();
        assert_eq!(
            memory.direct_protection_at(space, start),
            Some(DirectProtection::None)
        );
        memory
            .set_permissions(space, start, size, MemoryPermissions::READ)
            .unwrap();
        assert_eq!(
            memory.direct_protection_at(space, start),
            Some(DirectProtection::Read)
        );
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
