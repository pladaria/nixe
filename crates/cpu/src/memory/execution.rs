//! Production process-memory storage.
//!
//! [`ExecutionMemory`] deliberately does not reuse [`super::SyntheticMemory`].
//! The synthetic backend favors deterministic fault injection and simple
//! observability. This backend instead resolves a virtual page through one
//! sparse radix leaf and then indexes a stable physical-page slot directly.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};

use crate::{
    address::{AddressSpaceId, CodeGeneration, GuestPhysicalPageId, GuestVirtualAddress},
    error::{InstructionFetchFault, InstructionFetchFaultReason},
    vcpu::ExclusiveReservation,
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
    Ram {
        bytes: Option<Box<[u8; SYNTHETIC_PAGE_SIZE]>>,
        generation: u64,
    },
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
}

impl ExecutionMemoryInner {
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
            ExecutionPhysicalPage::Ram { .. } => MemoryRegionKind::Ram,
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
/// Interior mutability remains necessary because [`CpuMemory`] models guest
/// writes and MMIO through shared references. The `RefCell` protects the safe,
/// single-threaded ownership contract; resolved physical pages are nevertheless
/// reached by direct slot indexing rather than a second associative lookup.
pub struct ExecutionMemory {
    inner: RefCell<ExecutionMemoryInner>,
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
            ..ExecutionMemoryInner::default()
        };
        Self {
            inner: RefCell::new(inner),
        }
    }

    /// Atomically installs initialized RAM pages.
    pub fn install_ram_pages_atomic(
        &mut self,
        address_space: AddressSpaceId,
        requests: &[SyntheticRamPage<'_>],
    ) -> Result<(), SyntheticInstallError> {
        let inner = self.inner.get_mut();
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
            let mut contents = Box::new([0; SYNTHETIC_PAGE_SIZE]);
            contents.copy_from_slice(request.bytes);
            pending.push((
                virtual_pages[index],
                physical_page,
                request.permissions,
                ExecutionPhysicalPage::Ram {
                    bytes: Some(contents),
                    generation: 1,
                },
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
        self.inner
            .borrow()
            .mapping_at(address_space, virtual_address)
            .map(|mapping| SyntheticMappingInfo {
                physical_page: mapping.physical_page,
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
        let inner = self.inner.get_mut();
        if !(first_page..end_page).all(|page| inner.mappings.get(address_space, page).is_some()) {
            return false;
        }
        for page in first_page..end_page {
            inner
                .mappings
                .get_mut(address_space, page)
                .expect("range was preflighted")
                .purpose = purpose;
        }
        true
    }

    /// Returns the number of physical pages owned by this backend.
    #[must_use]
    pub fn physical_page_count(&self) -> usize {
        self.inner.borrow().slots_by_id.len()
    }

    /// Creates a zero-filled RAM page for explicit runtime or differential setup.
    pub fn add_ram_page(&mut self, page: GuestPhysicalPageId) -> bool {
        self.inner
            .get_mut()
            .push_page(
                page,
                ExecutionPhysicalPage::Ram {
                    bytes: Some(Box::new([0; SYNTHETIC_PAGE_SIZE])),
                    generation: 0,
                },
            )
            .is_some()
    }

    /// Creates a device-backed physical page.
    pub fn add_mmio_page(
        &mut self,
        page: GuestPhysicalPageId,
        handler: impl SyntheticMmio + 'static,
    ) -> bool {
        self.inner
            .get_mut()
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
        let inner = self.inner.get_mut();
        let Some(&slot) = inner.slots_by_id.get(&page) else {
            return false;
        };
        let Some(ExecutionPhysicalPage::Ram {
            bytes: contents,
            generation,
        }) = inner.page_mut(slot)
        else {
            return false;
        };
        let Some(end) = offset.checked_add(bytes.len()) else {
            return false;
        };
        let contents = contents.get_or_insert_with(|| Box::new([0; SYNTHETIC_PAGE_SIZE]));
        let Some(destination) = contents.get_mut(offset..end) else {
            return false;
        };
        destination.copy_from_slice(bytes);
        *generation = generation.wrapping_add(1);
        true
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
        let Some(end) = address.get().checked_add(bytes.len() as u64) else {
            return false;
        };
        let mut inner = self.inner.borrow_mut();
        let mut cursor = address.get();
        while cursor < end {
            let virtual_address = GuestVirtualAddress::new(cursor);
            let Some(mapping) = inner.mapping_at(address_space, virtual_address) else {
                return false;
            };
            if !matches!(
                inner.page(mapping.physical_slot),
                Some(ExecutionPhysicalPage::Ram { .. })
            ) {
                return false;
            }
            let remaining_in_page = SYNTHETIC_PAGE_SIZE - page_offset(virtual_address);
            cursor = cursor.saturating_add(remaining_in_page.min((end - cursor) as usize) as u64);
        }

        let mut copied = 0;
        while copied < bytes.len() {
            let virtual_address =
                GuestVirtualAddress::new(address.get().saturating_add(copied as u64));
            let mapping = inner
                .mapping_at(address_space, virtual_address)
                .expect("host overwrite range was validated");
            let offset = page_offset(virtual_address);
            let count = (SYNTHETIC_PAGE_SIZE - offset).min(bytes.len() - copied);
            let Some(ExecutionPhysicalPage::Ram {
                bytes: contents,
                generation,
            }) = inner.page_mut(mapping.physical_slot)
            else {
                unreachable!("host overwrite RAM range was validated")
            };
            contents.get_or_insert_with(|| Box::new([0; SYNTHETIC_PAGE_SIZE]))
                [offset..offset + count]
                .copy_from_slice(&bytes[copied..copied + count]);
            *generation = generation.wrapping_add(1);
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
        let inner = self.inner.get_mut();
        let Some(&physical_slot) = inner.slots_by_id.get(&physical_page) else {
            return false;
        };
        inner
            .mappings
            .insert(
                address_space,
                virtual_address.get() >> PAGE_SHIFT,
                ExecutionMapping {
                    physical_page,
                    physical_slot,
                    permissions,
                    purpose: MemoryMappingPurpose::Normal,
                    attributes: MemoryAttributes::NONE,
                },
            )
            .is_none()
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
        let inner = self.inner.borrow();
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
        let Some(ExecutionPhysicalPage::Ram {
            bytes: contents,
            generation,
        }) = inner.page(mapping.physical_slot)
        else {
            return Err(InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::Memory("executable mapping is not RAM".into()),
            ));
        };
        let mut bytes = [0; N];
        if let Some(contents) = contents {
            bytes.copy_from_slice(&contents[page_offset(address)..end_offset]);
        }
        Ok((
            bytes,
            CodeDependencies::one(CodePageDependency {
                page: mapping.physical_page,
                generation: CodeGeneration::new(*generation),
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
        let inner = self.inner.borrow();
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
            Some(ExecutionPhysicalPage::Ram { .. }) => MemoryRegionKind::Ram,
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
        let mut inner = self.inner.borrow_mut();
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
                let ExecutionPhysicalPage::Ram {
                    bytes: contents, ..
                } = inner
                    .page(resolved.first.physical_slot)
                    .expect("resolved RAM page exists")
                else {
                    unreachable!()
                };
                if let Some(contents) = contents {
                    let offset = page_offset(address);
                    bytes[..byte_count].copy_from_slice(&contents[offset..offset + byte_count]);
                }
            }
            Some(second) => {
                let mappings = [resolved.first, second];
                let mut copied = 0;
                for (mapping, count, offset) in [
                    (mappings[0], resolved.first_bytes, page_offset(address)),
                    (mappings[1], byte_count - resolved.first_bytes, 0),
                ] {
                    let ExecutionPhysicalPage::Ram {
                        bytes: contents, ..
                    } = inner
                        .page(mapping.physical_slot)
                        .expect("resolved RAM page exists")
                    else {
                        unreachable!()
                    };
                    if let Some(contents) = contents {
                        bytes[copied..copied + count]
                            .copy_from_slice(&contents[offset..offset + count]);
                    }
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
        let mut inner = self.inner.borrow_mut();
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
                let ExecutionPhysicalPage::Ram {
                    bytes: contents,
                    generation,
                } = inner
                    .page_mut(resolved.first.physical_slot)
                    .expect("resolved RAM page exists")
                else {
                    unreachable!()
                };
                let contents = contents.get_or_insert_with(|| Box::new([0; SYNTHETIC_PAGE_SIZE]));
                let offset = page_offset(address);
                contents[offset..offset + byte_count].copy_from_slice(&bytes[..byte_count]);
                *generation = generation.wrapping_add(1);
            }
            Some(second) => {
                let mappings = [resolved.first, second];
                let mut copied = 0;
                for (mapping, count, offset) in [
                    (mappings[0], resolved.first_bytes, page_offset(address)),
                    (mappings[1], byte_count - resolved.first_bytes, 0),
                ] {
                    let ExecutionPhysicalPage::Ram {
                        bytes: contents, ..
                    } = inner
                        .page_mut(mapping.physical_slot)
                        .expect("resolved RAM page exists")
                    else {
                        unreachable!()
                    };
                    let contents =
                        contents.get_or_insert_with(|| Box::new([0; SYNTHETIC_PAGE_SIZE]));
                    contents[offset..offset + count]
                        .copy_from_slice(&bytes[copied..copied + count]);
                    copied += count;
                }
                for slot in [
                    Some(mappings[0].physical_slot),
                    (mappings[1].physical_slot != mappings[0].physical_slot)
                        .then_some(mappings[1].physical_slot),
                ]
                .into_iter()
                .flatten()
                {
                    let Some(ExecutionPhysicalPage::Ram { generation, .. }) = inner.page_mut(slot)
                    else {
                        unreachable!()
                    };
                    *generation = generation.wrapping_add(1);
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
        let inner = self.inner.borrow();
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
        let value = self.read(address_space, address, access)?;
        let inner = self.inner.borrow();
        let mapping = inner
            .mapping_at(address_space, address)
            .expect("load was validated");
        let ExecutionPhysicalPage::Ram { generation, .. } = inner
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
        Ok((
            value,
            ExclusiveReservation {
                page: mapping.physical_page,
                byte_offset: page_offset(address) as u16,
                access_size: access.size.bytes() as u8,
                generation: CodeGeneration::new(*generation),
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
        let matches = {
            let inner = self.inner.borrow();
            let resolved = resolve_data_access(
                &inner,
                address_space,
                address,
                access,
                DataAccessKind::Write,
            )?;
            let generation = match inner.page(resolved.first.physical_slot) {
                Some(ExecutionPhysicalPage::Ram { generation, .. }) => *generation,
                _ => {
                    return Err(DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::MixedRegions,
                    ));
                }
            };
            reservation.page == resolved.first.physical_page
                && usize::from(reservation.byte_offset) == page_offset(address)
                && usize::from(reservation.access_size) == access.size.bytes()
                && reservation.generation == CodeGeneration::new(generation)
        };
        if !matches {
            return Ok((
                DataWriteResult {
                    region: MemoryRegionKind::Ram,
                },
                false,
            ));
        }
        self.write(address_space, address, access, value)
            .map(|result| (result, true))
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
        let mut inner = self.inner.borrow_mut();
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
            return Ok(());
        }

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
            pending.push((page, physical_page));
        }
        for (page, physical_page) in pending {
            let slot = inner
                .push_page(
                    physical_page,
                    ExecutionPhysicalPage::Ram {
                        bytes: None,
                        generation: 1,
                    },
                )
                .expect("allocated physical identity is unique");
            let previous = inner.mappings.insert(
                address_space,
                page,
                ExecutionMapping {
                    physical_page,
                    physical_slot: slot,
                    permissions,
                    purpose,
                    attributes: MemoryAttributes::NONE,
                },
            );
            debug_assert!(previous.is_none());
        }
        inner.next_page_id = next_page_id;
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
        let mut inner = self.inner.borrow_mut();
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
        let mut changed_executable_slots = BTreeSet::new();
        for page in first_page..end_page {
            let mapping = inner
                .mappings
                .get_mut(address_space, page)
                .expect("protection range was preflighted");
            if mapping.permissions.contains(MemoryPermissions::EXECUTE)
                != permissions.contains(MemoryPermissions::EXECUTE)
            {
                changed_executable_slots.insert(mapping.physical_slot);
            }
            mapping.permissions = permissions;
        }
        for slot in changed_executable_slots {
            if let Some(ExecutionPhysicalPage::Ram { generation, .. }) = inner.page_mut(slot) {
                *generation = generation.wrapping_add(1);
            }
        }
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
        let mut inner = self.inner.borrow_mut();
        for page in first_page..end_page {
            if inner.mappings.get(address_space, page).is_none() {
                return Err(error(
                    GuestVirtualAddress::new(page * page_size),
                    MemoryProtectionErrorReason::Unmapped,
                ));
            }
        }
        for page in first_page..end_page {
            let mapping = inner
                .mappings
                .get_mut(address_space, page)
                .expect("attribute range was preflighted");
            let bits = (mapping.attributes.bits() & !mask.bits()) | value.bits();
            mapping.attributes = MemoryAttributes::from_bits(bits)
                .expect("existing, masked, and replacement attributes are bounded");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let inner = memory.inner.get_mut();
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

        let inner = memory.inner.borrow();
        assert_eq!(inner.physical_slots.len(), 4);
        assert_eq!(inner.slots_by_id.len(), 4);
        assert!(inner.free_physical_slots.is_empty());
    }

    #[test]
    fn identity_exhaustion_does_not_publish_partial_installation() {
        let mut memory = ExecutionMemory::new();
        memory.inner.get_mut().next_page_id = u64::MAX;
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
