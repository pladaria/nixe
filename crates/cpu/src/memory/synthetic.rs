//! Deterministic memory backend for frontend and runtime tests.

use nixe_memory::{
    AddressSpaceId, ContentGeneration, ContentMutationEpoch, GuestPhysicalPageId,
    GuestVirtualAddress, MappingGeneration, MemoryInvalidation, MemoryInvalidationCursor,
    MemoryInvalidationError, MemoryInvalidationKind, MemoryInvalidationLog,
    MemoryInvalidationSource,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Mutex, MutexGuard, PoisonError},
};

use crate::error::{InstructionFetchFault, InstructionFetchFaultReason};

use super::common::{
    MappingState, PAGE_SIZE, PageRange, ResolvedDataAccess, allocate_page_id,
    coalesce_mapped_pages, install_error, masked_attributes, memory_query_result, page_address,
    page_offset, resolve_data_access, take_mapping_generation, validate_install_request,
    virtual_page, writable_executable,
};
use super::{
    AtomicMemoryResult, AtomicRmwKind, CodeDependencies, CodePageDependency, CodePageSpan,
    CpuMemory, DataAccessFault, DataAccessFaultReason, DataAccessKind, DataReadResult,
    DataWriteResult, FetchedCode, InstructionMemory, MemoryAccess, MemoryAccessClass,
    MemoryAlignment, MemoryAttributes, MemoryMappingError, MemoryMappingErrorReason,
    MemoryMappingPurpose, MemoryPermissions, MemoryProtectionError, MemoryProtectionErrorReason,
    MemoryQueryResult, MemoryRegionKind, MemoryValue, ProcessMemory, SYNTHETIC_PAGE_SIZE,
    SyntheticInstallError, SyntheticInstallStage, SyntheticMappingInfo, SyntheticMmio,
    SyntheticRamPage,
};

#[derive(Clone, Copy)]
struct Mapping {
    physical_page: GuestPhysicalPageId,
    mapping_generation: MappingGeneration,
    permissions: MemoryPermissions,
    purpose: MemoryMappingPurpose,
    attributes: MemoryAttributes,
}

enum PhysicalPage {
    Ram {
        bytes: Option<Box<[u8; SYNTHETIC_PAGE_SIZE]>>,
        generation: ContentGeneration,
    },
    Mmio(Box<dyn SyntheticMmio>),
}

struct SyntheticMemoryInner {
    mappings: BTreeMap<(AddressSpaceId, u64), Mapping>,
    pages: BTreeMap<GuestPhysicalPageId, PhysicalPage>,
    instruction_faults: BTreeMap<(AddressSpaceId, GuestVirtualAddress), Box<str>>,
    data_faults: BTreeMap<(AddressSpaceId, GuestVirtualAddress, DataAccessKind), Box<str>>,
    next_page_id: u64,
    next_mapping_generation: Option<MappingGeneration>,
    install_failure: Option<(SyntheticInstallStage, usize, Box<str>)>,
    content_epoch: ContentMutationEpoch,
}

impl Default for SyntheticMemoryInner {
    fn default() -> Self {
        Self {
            mappings: BTreeMap::new(),
            pages: BTreeMap::new(),
            instruction_faults: BTreeMap::new(),
            data_faults: BTreeMap::new(),
            next_page_id: 1,
            next_mapping_generation: Some(MappingGeneration::new(1)),
            install_failure: None,
            content_epoch: ContentMutationEpoch::INITIAL,
        }
    }
}

/// Small deterministic process-memory implementation for frontend tests.
///
/// Its APIs expose copies, identities, and callbacks only; no raw mutable host
/// pointer crosses the CPU/memory boundary.
#[derive(Default)]
pub struct SyntheticMemory {
    inner: Mutex<SyntheticMemoryInner>,
    invalidations: MemoryInvalidationLog,
}

impl SyntheticMemory {
    /// Creates empty synthetic memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock_inner(&self) -> MutexGuard<'_, SyntheticMemoryInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn inner_mut(&mut self) -> &mut SyntheticMemoryInner {
        self.inner.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    fn executable_page(
        inner: &SyntheticMemoryInner,
        page: GuestPhysicalPageId,
    ) -> Option<GuestPhysicalPageId> {
        inner
            .mappings
            .values()
            .any(|mapping| {
                mapping.physical_page == page
                    && (mapping.permissions.contains(MemoryPermissions::EXECUTE)
                        || mapping.purpose.is_code())
            })
            .then_some(page)
    }

    fn atomic_transaction(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        operation: impl FnOnce(MemoryValue) -> (MemoryValue, bool),
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
        let mut inner = self.lock_inner();
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
        for kind in [DataAccessKind::Read, DataAccessKind::Write] {
            if let Some(reason) = inner.data_faults.get(&(address_space, address, kind)) {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    kind,
                    DataAccessFaultReason::Injected(reason.clone()),
                ));
            }
        }
        let page = resolved.first.physical_page;
        let offset = page_offset(address);
        let byte_count = access.size.bytes();
        let mut previous_bytes = [0_u8; 16];
        let PhysicalPage::Ram { bytes, .. } = inner
            .pages
            .get(&page)
            .expect("resolved atomic RAM page exists")
        else {
            unreachable!()
        };
        if let Some(bytes) = bytes {
            previous_bytes[..byte_count].copy_from_slice(&bytes[offset..offset + byte_count]);
        }
        let previous = MemoryValue::from_le_slice(access.size, &previous_bytes[..byte_count]);
        let (replacement, stored) = operation(previous);
        if replacement.size() != access.size {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ValueSizeMismatch,
            ));
        }
        if stored {
            super::contracts::begin_ordered_write(access.ordering);
            let next_content_epoch = inner.content_epoch.next().map_err(|_| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::ContentGenerationExhausted,
                )
            })?;
            let invalidation = Self::executable_page(&inner, page)
                .map(|first| {
                    self.invalidations
                        .reserve(MemoryInvalidationKind::ExecutableContent {
                            first,
                            second: None,
                        })
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
            let PhysicalPage::Ram { bytes, generation } = inner
                .pages
                .get_mut(&page)
                .expect("resolved atomic RAM page exists")
            else {
                unreachable!()
            };
            let next_generation = generation.next().map_err(|_| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::ContentGenerationExhausted,
                )
            })?;
            let bytes = bytes.get_or_insert_with(|| Box::new([0; SYNTHETIC_PAGE_SIZE]));
            replacement.copy_le_bytes(&mut bytes[offset..offset + byte_count]);
            *generation = next_generation;
            inner.content_epoch = next_content_epoch;
            if let Some(invalidation) = invalidation {
                invalidation.commit();
            }
        }
        super::contracts::complete_ordered_read(access.ordering);
        Ok(AtomicMemoryResult {
            previous,
            stored,
            region: MemoryRegionKind::Ram,
        })
    }

    /// Atomically creates, initializes, and publishes ordinary RAM pages.
    ///
    /// Physical pages are owned by this memory object after success. A failed
    /// request changes neither existing mappings nor physical pages.
    pub fn install_ram_pages_atomic(
        &mut self,
        address_space: AddressSpaceId,
        requests: &[SyntheticRamPage<'_>],
    ) -> Result<(), SyntheticInstallError> {
        let invalidations = &self.invalidations;
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        let mut virtual_pages = Vec::with_capacity(requests.len());
        let mut unique_virtual_pages = BTreeSet::new();
        for (index, request) in requests.iter().enumerate() {
            fail_install_if_requested(
                inner,
                SyntheticInstallStage::Preflight,
                index,
                request.virtual_address,
            )?;
            let virtual_page = validate_install_request(*request, &mut unique_virtual_pages)?;
            if inner.mappings.contains_key(&(address_space, virtual_page)) {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "virtual page is already mapped",
                ));
            }
            virtual_pages.push(virtual_page);
        }

        let mut next_page_id = inner.next_page_id;
        let mut pending = Vec::with_capacity(requests.len());
        for (index, request) in requests.iter().enumerate() {
            fail_install_if_requested(
                inner,
                SyntheticInstallStage::Allocation,
                index,
                request.virtual_address,
            )?;
            let physical_page =
                allocate_page_id(&mut next_page_id, |page| inner.pages.contains_key(&page))
                    .ok_or_else(|| {
                        install_error(
                            SyntheticInstallStage::Allocation,
                            Some(request.virtual_address),
                            "physical-page identities are exhausted",
                        )
                    })?;
            fail_install_if_requested(
                inner,
                SyntheticInstallStage::Initialization,
                index,
                request.virtual_address,
            )?;
            let mut contents = Box::new([0; SYNTHETIC_PAGE_SIZE]);
            contents.copy_from_slice(request.bytes);
            pending.push((
                virtual_pages[index],
                Mapping {
                    physical_page,
                    mapping_generation: MappingGeneration::INITIAL,
                    permissions: request.permissions,
                    purpose: MemoryMappingPurpose::Normal,
                    attributes: MemoryAttributes::NONE,
                },
                PhysicalPage::Ram {
                    bytes: Some(contents),
                    generation: ContentGeneration::new(1),
                },
            ));
        }
        for (index, request) in requests.iter().enumerate() {
            fail_install_if_requested(
                inner,
                SyntheticInstallStage::Publication,
                index,
                request.virtual_address,
            )?;
        }

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
        let invalidation = (!invalidation_kinds.is_empty())
            .then(|| invalidations.reserve_many(&invalidation_kinds))
            .transpose()
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
        for (_, mapping, _) in &mut pending {
            mapping.mapping_generation = mapping_generation;
        }
        for (virtual_page, mapping, page) in pending {
            let previous_page = inner.pages.insert(mapping.physical_page, page);
            let previous_mapping = inner
                .mappings
                .insert((address_space, virtual_page), mapping);
            debug_assert!(previous_page.is_none());
            debug_assert!(previous_mapping.is_none());
        }
        inner.next_page_id = next_page_id;
        if let Some(invalidation) = invalidation {
            invalidation.commit();
        }
        Ok(())
    }

    /// Injects a deterministic failure into a future atomic installation.
    pub fn inject_install_failure(
        &mut self,
        stage: SyntheticInstallStage,
        request_index: usize,
        reason: impl Into<Box<str>>,
    ) {
        self.inner_mut().install_failure = Some((stage, request_index, reason.into()));
    }

    /// Returns mapping identity and permissions for a page containing `address`.
    pub fn mapping_info(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Option<SyntheticMappingInfo> {
        mapping_at(&self.lock_inner(), address_space, address).map(|mapping| SyntheticMappingInfo {
            physical_page: mapping.physical_page,
            mapping_generation: mapping.mapping_generation,
            permissions: mapping.permissions,
            attributes: mapping.attributes,
            purpose: mapping.purpose,
        })
    }

    /// Assigns a semantic purpose to an already mapped page-aligned range.
    ///
    /// The operation validates the complete range before changing any page.
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
        if !(range.first..range.end).all(|page| inner.mappings.contains_key(&(address_space, page)))
        {
            return false;
        }
        if (range.first..range.end)
            .all(|page| inner.mappings[&(address_space, page)].purpose == purpose)
        {
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
            let mapping = inner
                .mappings
                .get_mut(&(address_space, page))
                .expect("range was preflighted");
            mapping.purpose = purpose;
            mapping.mapping_generation = mapping_generation;
        }
        invalidation.commit();
        true
    }

    /// Returns the number of physical pages currently owned by this backend.
    pub fn physical_page_count(&self) -> usize {
        self.lock_inner().pages.len()
    }

    /// Creates a zero-filled ordinary physical page.
    pub fn add_ram_page(&mut self, page: GuestPhysicalPageId) -> bool {
        self.inner_mut()
            .pages
            .insert(
                page,
                PhysicalPage::Ram {
                    bytes: Some(Box::new([0; SYNTHETIC_PAGE_SIZE])),
                    generation: ContentGeneration::INITIAL,
                },
            )
            .is_none()
    }

    /// Creates a device-backed physical page.
    pub fn add_mmio_page(
        &mut self,
        page: GuestPhysicalPageId,
        handler: impl SyntheticMmio + 'static,
    ) -> bool {
        self.inner_mut()
            .pages
            .insert(page, PhysicalPage::Mmio(Box::new(handler)))
            .is_none()
    }

    /// Maps one page-aligned virtual page; aliases map the same physical ID again.
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
        if !inner.pages.contains_key(&physical_page) {
            return false;
        }
        let key = (address_space, virtual_page(virtual_address));
        if inner.mappings.contains_key(&key) {
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
        let previous = inner.mappings.insert(
            key,
            Mapping {
                physical_page,
                mapping_generation,
                permissions,
                purpose: MemoryMappingPurpose::Normal,
                attributes: MemoryAttributes::NONE,
            },
        );
        debug_assert!(previous.is_none());
        invalidation.commit();
        true
    }

    /// Copies fixture bytes directly into a RAM page and advances its generation.
    pub fn initialize_ram(
        &mut self,
        page: GuestPhysicalPageId,
        offset: usize,
        bytes: &[u8],
    ) -> bool {
        let invalidations = &self.invalidations;
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        let Ok(next_epoch) = inner.content_epoch.next() else {
            return false;
        };
        let code_observed = inner.mappings.values().any(|mapping| {
            mapping.physical_page == page
                && (mapping.permissions.contains(MemoryPermissions::EXECUTE)
                    || mapping.purpose.is_code())
        });
        let Some(PhysicalPage::Ram {
            bytes: contents,
            generation,
        }) = inner.pages.get_mut(&page)
        else {
            return false;
        };
        let Some(end) = offset.checked_add(bytes.len()) else {
            return false;
        };
        let invalidation = if code_observed {
            match invalidations.reserve(MemoryInvalidationKind::ExecutableContent {
                first: page,
                second: None,
            }) {
                Ok(invalidation) => Some(invalidation),
                Err(_) => return false,
            }
        } else {
            None
        };
        let contents = contents.get_or_insert_with(|| Box::new([0; SYNTHETIC_PAGE_SIZE]));
        let Some(destination) = contents.get_mut(offset..end) else {
            return false;
        };
        let Ok(next_generation) = generation.next() else {
            return false;
        };
        destination.copy_from_slice(bytes);
        *generation = next_generation;
        inner.content_epoch = next_epoch;
        if let Some(invalidation) = invalidation {
            invalidation.commit();
        }
        true
    }

    /// Injects a deterministic fetch failure at an exact virtual address.
    pub fn inject_instruction_fault(
        &mut self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        reason: impl Into<Box<str>>,
    ) {
        self.inner_mut()
            .instruction_faults
            .insert((address_space, address), reason.into());
    }

    /// Injects a deterministic data failure at an exact operation address.
    pub fn inject_data_fault(
        &mut self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        kind: DataAccessKind,
        reason: impl Into<Box<str>>,
    ) {
        self.inner_mut()
            .data_faults
            .insert((address_space, address, kind), reason.into());
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
        if end_offset <= SYNTHETIC_PAGE_SIZE {
            if !inner.instruction_faults.is_empty()
                && let Some((fault_address, reason)) = (0..N).find_map(|index| {
                    let current = address.checked_add(index as u64)?;
                    inner
                        .instruction_faults
                        .get(&(address_space, current))
                        .map(|reason| (current, reason))
                })
            {
                return Err(InstructionFetchFault::new(
                    address_space,
                    fault_address,
                    InstructionFetchFaultReason::Memory(reason.clone()),
                ));
            }
            let mapping = mapping_at(&inner, address_space, address).ok_or_else(|| {
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
            let Some(PhysicalPage::Ram {
                bytes: contents,
                generation,
            }) = inner.pages.get(&mapping.physical_page)
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
            return Ok((
                bytes,
                CodeDependencies::one(CodePageDependency {
                    page: mapping.physical_page,
                    generation: *generation,
                    mapping_generation: mapping.mapping_generation,
                }),
            ));
        }

        let mut bytes = [0; N];
        let mut dependencies: Option<CodeDependencies> = None;
        for (index, destination) in bytes.iter_mut().enumerate() {
            let Some(current) = address.checked_add(index as u64) else {
                return Err(InstructionFetchFault::new(
                    address_space,
                    address,
                    InstructionFetchFaultReason::AddressOverflow,
                ));
            };
            if let Some(reason) = inner.instruction_faults.get(&(address_space, current)) {
                return Err(InstructionFetchFault::new(
                    address_space,
                    current,
                    InstructionFetchFaultReason::Memory(reason.clone()),
                ));
            }
            let mapping = mapping_at(&inner, address_space, current).ok_or_else(|| {
                InstructionFetchFault::new(
                    address_space,
                    current,
                    InstructionFetchFaultReason::Unmapped,
                )
            })?;
            if !mapping.permissions.contains(MemoryPermissions::EXECUTE) {
                return Err(InstructionFetchFault::new(
                    address_space,
                    current,
                    InstructionFetchFaultReason::ExecutePermissionDenied,
                ));
            }
            let Some(PhysicalPage::Ram {
                bytes: contents,
                generation,
            }) = inner.pages.get(&mapping.physical_page)
            else {
                return Err(InstructionFetchFault::new(
                    address_space,
                    current,
                    InstructionFetchFaultReason::Memory("executable mapping is not RAM".into()),
                ));
            };
            *destination = contents
                .as_ref()
                .map_or(0, |contents| contents[page_offset(current)]);
            let dependency = CodePageDependency {
                page: mapping.physical_page,
                generation: *generation,
                mapping_generation: mapping.mapping_generation,
            };
            dependencies = Some(match dependencies {
                None => CodeDependencies::one(dependency),
                Some(current_dependencies) => {
                    current_dependencies.merge(CodeDependencies::one(dependency))
                }
            });
        }
        Ok((
            bytes,
            dependencies.expect("non-empty fetch has a dependency"),
        ))
    }
}

impl MemoryInvalidationSource for SyntheticMemory {
    fn invalidation_cursor(&self) -> MemoryInvalidationCursor {
        self.invalidations.cursor()
    }

    fn read_invalidations_since(
        &self,
        after: MemoryInvalidationCursor,
        output: &mut Vec<MemoryInvalidation>,
    ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError> {
        self.invalidations.read_since(after, output)
    }
}

fn fail_install_if_requested(
    inner: &SyntheticMemoryInner,
    stage: SyntheticInstallStage,
    index: usize,
    address: GuestVirtualAddress,
) -> Result<(), SyntheticInstallError> {
    if let Some((requested_stage, requested_index, reason)) = &inner.install_failure
        && *requested_stage == stage
        && *requested_index == index
    {
        return Err(install_error(stage, Some(address), reason.clone()));
    }
    Ok(())
}

impl InstructionMemory for SyntheticMemory {
    fn content_mutation_epoch(&self) -> ContentMutationEpoch {
        self.lock_inner().content_epoch
    }

    fn code_page_span(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<CodePageSpan, InstructionFetchFault> {
        let page_start = page_address(virtual_page(address));
        let inner = self.lock_inner();
        let mapping = mapping_at(&inner, address_space, address).ok_or_else(|| {
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
            .expect("synthetic page arithmetic contains its source address"))
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

impl CpuMemory for SyntheticMemory {
    fn read(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault> {
        let mut inner = self.lock_inner();
        let resolved =
            resolve_access(&inner, address_space, address, access, DataAccessKind::Read)?;
        if let Some(reason) = inner
            .data_faults
            .get(&(address_space, address, DataAccessKind::Read))
        {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::Injected(reason.clone()),
            ));
        }
        if resolved.region == MemoryRegionKind::Device {
            if resolved.second.is_some() {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Read,
                    DataAccessFaultReason::MixedRegions,
                ));
            }
            let PhysicalPage::Mmio(handler) = inner
                .pages
                .get_mut(&resolved.first.physical_page)
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
                let PhysicalPage::Ram {
                    bytes: contents, ..
                } = inner
                    .pages
                    .get(&resolved.first.physical_page)
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
                    let PhysicalPage::Ram {
                        bytes: contents, ..
                    } = inner
                        .pages
                        .get(&mapping.physical_page)
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
        if let Some(reason) =
            inner
                .data_faults
                .get(&(address_space, address, DataAccessKind::Write))
        {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::Injected(reason.clone()),
            ));
        }
        if resolved.region == MemoryRegionKind::Device {
            if resolved.second.is_some() {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::MixedRegions,
                ));
            }
            let PhysicalPage::Mmio(handler) = inner
                .pages
                .get_mut(&resolved.first.physical_page)
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
        let next_content_epoch = inner.content_epoch.next().map_err(|_| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ContentGenerationExhausted,
            )
        })?;
        let first_code_page = Self::executable_page(&inner, resolved.first.physical_page);
        let second_code_page = resolved
            .second
            .and_then(|second| Self::executable_page(&inner, second.physical_page))
            .filter(|second| Some(*second) != first_code_page);
        let invalidation = first_code_page
            .or(second_code_page)
            .map(|first| {
                self.invalidations
                    .reserve(MemoryInvalidationKind::ExecutableContent {
                        first,
                        second: second_code_page,
                    })
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
        let byte_count = access.size.bytes();
        let mut bytes = [0_u8; 16];
        value.copy_le_bytes(&mut bytes[..byte_count]);
        super::contracts::begin_ordered_write(access.ordering);
        match resolved.second {
            None => {
                let PhysicalPage::Ram {
                    bytes: contents,
                    generation,
                } = inner
                    .pages
                    .get_mut(&resolved.first.physical_page)
                    .expect("resolved RAM page exists")
                else {
                    unreachable!()
                };
                let next_generation = generation.next().map_err(|_| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::ContentGenerationExhausted,
                    )
                })?;
                let contents = contents.get_or_insert_with(|| Box::new([0; SYNTHETIC_PAGE_SIZE]));
                let offset = page_offset(address);
                contents[offset..offset + byte_count].copy_from_slice(&bytes[..byte_count]);
                *generation = next_generation;
            }
            Some(second) => {
                let mappings = [resolved.first, second];
                let next_first = match inner.pages.get(&mappings[0].physical_page) {
                    Some(PhysicalPage::Ram { generation, .. }) => {
                        generation.next().map_err(|_| {
                            DataAccessFault::new(
                                address_space,
                                address,
                                DataAccessKind::Write,
                                DataAccessFaultReason::ContentGenerationExhausted,
                            )
                        })?
                    }
                    _ => unreachable!("resolved RAM page exists"),
                };
                let next_second = if mappings[1].physical_page != mappings[0].physical_page {
                    match inner.pages.get(&mappings[1].physical_page) {
                        Some(PhysicalPage::Ram { generation, .. }) => {
                            Some(generation.next().map_err(|_| {
                                DataAccessFault::new(
                                    address_space,
                                    address,
                                    DataAccessKind::Write,
                                    DataAccessFaultReason::ContentGenerationExhausted,
                                )
                            })?)
                        }
                        _ => unreachable!("resolved RAM page exists"),
                    }
                } else {
                    None
                };
                let mut copied = 0;
                for (mapping, count, offset) in [
                    (mappings[0], resolved.first_bytes, page_offset(address)),
                    (mappings[1], byte_count - resolved.first_bytes, 0),
                ] {
                    let PhysicalPage::Ram {
                        bytes: contents, ..
                    } = inner
                        .pages
                        .get_mut(&mapping.physical_page)
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
                let Some(PhysicalPage::Ram { generation, .. }) =
                    inner.pages.get_mut(&mappings[0].physical_page)
                else {
                    unreachable!()
                };
                *generation = next_first;
                if let Some(next_second) = next_second {
                    let Some(PhysicalPage::Ram { generation, .. }) =
                        inner.pages.get_mut(&mappings[1].physical_page)
                    else {
                        unreachable!()
                    };
                    *generation = next_second;
                }
            }
        }
        inner.content_epoch = next_content_epoch;
        if let Some(invalidation) = invalidation {
            invalidation.commit();
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
        let mapping = mapping_at(&inner, address_space, address).ok_or_else(|| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::Unmapped,
            )
        })?;
        if !matches!(
            inner.pages.get(&mapping.physical_page),
            Some(PhysicalPage::Ram { .. })
        ) {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::Device("cache maintenance requires RAM".into()),
            ));
        }
        if kind == super::CacheMaintenanceKind::InstructionInvalidate {
            self.invalidations
                .reserve(MemoryInvalidationKind::ExecutableContent {
                    first: mapping.physical_page,
                    second: None,
                })
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
        Ok(())
    }

    fn load_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<(DataReadResult, crate::exclusive::ExclusiveReservation), DataAccessFault> {
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
        let PhysicalPage::Ram { bytes, generation } = inner
            .pages
            .get(&mapping.physical_page)
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
        let offset = page_offset(address);
        let mut value_bytes = [0_u8; 16];
        if let Some(bytes) = bytes {
            value_bytes[..byte_count].copy_from_slice(&bytes[offset..offset + byte_count]);
        }
        super::contracts::complete_ordered_read(access.ordering);
        Ok((
            DataReadResult {
                value: MemoryValue::from_le_slice(access.size, &value_bytes[..byte_count]),
                region: MemoryRegionKind::Ram,
            },
            crate::exclusive::ExclusiveReservation {
                page: mapping.physical_page,
                byte_offset: page_offset(address) as u16,
                access_size: access.size.bytes() as u8,
                generation: *generation,
            },
        ))
    }

    fn store_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
        reservation: crate::exclusive::ExclusiveReservation,
    ) -> Result<(DataWriteResult, bool), DataAccessFault> {
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
        if resolved.second.is_some() {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::MixedRegions,
            ));
        }
        let generation = match inner.pages.get(&resolved.first.physical_page) {
            Some(PhysicalPage::Ram { generation, .. }) => *generation,
            _ => {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::MixedRegions,
                ));
            }
        };
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
        let next_content_epoch = inner.content_epoch.next().map_err(|_| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ContentGenerationExhausted,
            )
        })?;
        let invalidation = Self::executable_page(&inner, resolved.first.physical_page)
            .map(|first| {
                self.invalidations
                    .reserve(MemoryInvalidationKind::ExecutableContent {
                        first,
                        second: None,
                    })
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
        let PhysicalPage::Ram {
            bytes: contents,
            generation,
        } = inner
            .pages
            .get_mut(&resolved.first.physical_page)
            .expect("resolved exclusive RAM page exists")
        else {
            unreachable!()
        };
        let next_generation = generation.next().map_err(|_| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ContentGenerationExhausted,
            )
        })?;
        let contents = contents.get_or_insert_with(|| Box::new([0; SYNTHETIC_PAGE_SIZE]));
        let byte_count = access.size.bytes();
        let offset = page_offset(address);
        super::contracts::begin_ordered_write(access.ordering);
        value.copy_le_bytes(&mut contents[offset..offset + byte_count]);
        *generation = next_generation;
        inner.content_epoch = next_content_epoch;
        if let Some(invalidation) = invalidation {
            invalidation.commit();
        }
        Ok((
            DataWriteResult {
                region: MemoryRegionKind::Ram,
            },
            true,
        ))
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
        let state = synthetic_mapping_state(&inner, address_space, page);

        let (first_page, last_page_exclusive) = if let Some(state) = state {
            coalesce_mapped_pages(page, end_page, state, |page| {
                synthetic_mapping_state(&inner, address_space, page)
            })
        } else {
            let previous = inner
                .mappings
                .range(..(address_space, page))
                .next_back()
                .filter(|((space, _), _)| *space == address_space)
                .map_or(0, |((_, mapped_page), _)| mapped_page.saturating_add(1));
            let next = inner
                .mappings
                .range((address_space, page.saturating_add(1))..)
                .next()
                .filter(|((space, _), _)| *space == address_space)
                .map_or(end_page, |((_, mapped_page), _)| *mapped_page)
                .min(end_page);
            (previous.min(page), next.max(page + 1))
        };
        memory_query_result(first_page, last_page_exclusive, state)
    }
}

impl ProcessMemory for SyntheticMemory {
    fn read_bytes(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        output: &mut [u8],
    ) -> Result<(), DataAccessFault> {
        let size = u64::try_from(output.len()).map_err(|_| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::AddressOverflow,
            )
        })?;
        let end = address.get().checked_add(size).ok_or_else(|| {
            DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::AddressOverflow,
            )
        })?;
        let inner = self.lock_inner();
        if let Some(((.., fault_address, _), reason)) = inner
            .data_faults
            .iter()
            .filter(|((space, current, kind), _)| {
                *space == address_space
                    && *kind == DataAccessKind::Read
                    && current.get() >= address.get()
                    && current.get() < end
            })
            .min_by_key(|((_, current, _), _)| *current)
        {
            return Err(DataAccessFault::new(
                address_space,
                *fault_address,
                DataAccessKind::Read,
                DataAccessFaultReason::Injected(reason.clone()),
            ));
        }
        let mut copied = 0;
        while copied < output.len() {
            let current = address
                .checked_add(copied as u64)
                .expect("bulk read range was checked");
            let mapping = mapping_at(&inner, address_space, current).ok_or_else(|| {
                DataAccessFault::new(
                    address_space,
                    current,
                    DataAccessKind::Read,
                    DataAccessFaultReason::Unmapped,
                )
            })?;
            if !mapping.permissions.contains(MemoryPermissions::READ) {
                return Err(DataAccessFault::new(
                    address_space,
                    current,
                    DataAccessKind::Read,
                    DataAccessFaultReason::ReadPermissionDenied,
                ));
            }
            let PhysicalPage::Ram { bytes, .. } = inner
                .pages
                .get(&mapping.physical_page)
                .expect("mapped physical page exists")
            else {
                return Err(DataAccessFault::new(
                    address_space,
                    current,
                    DataAccessKind::Read,
                    DataAccessFaultReason::Device("bulk guest-memory reads require RAM".into()),
                ));
            };
            let offset = page_offset(current);
            let count = (SYNTHETIC_PAGE_SIZE - offset).min(output.len() - copied);
            if let Some(bytes) = bytes {
                output[copied..copied + count].copy_from_slice(&bytes[offset..offset + count]);
            } else {
                output[copied..copied + count].fill(0);
            }
            copied += count;
        }
        Ok(())
    }

    fn write_bytes(
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
        let mut inner = self.lock_inner();
        let next_content_epoch = (!bytes.is_empty())
            .then(|| inner.content_epoch.next())
            .transpose()
            .map_err(|_| {
                DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::ContentGenerationExhausted,
                )
            })?;
        if let Some(((.., fault_address, _), reason)) = inner
            .data_faults
            .iter()
            .filter(|((space, current, kind), _)| {
                *space == address_space
                    && *kind == DataAccessKind::Write
                    && current.get() >= address.get()
                    && current.get() < end
            })
            .min_by_key(|((_, current, _), _)| *current)
        {
            return Err(DataAccessFault::new(
                address_space,
                *fault_address,
                DataAccessKind::Write,
                DataAccessFaultReason::Injected(reason.clone()),
            ));
        }
        let mut next_generations = BTreeMap::new();
        let mut cursor = address.get();
        while cursor < end {
            let current = GuestVirtualAddress::new(cursor);
            let mapping = mapping_at(&inner, address_space, current).ok_or_else(|| {
                DataAccessFault::new(
                    address_space,
                    current,
                    DataAccessKind::Write,
                    DataAccessFaultReason::Unmapped,
                )
            })?;
            if !mapping.permissions.contains(MemoryPermissions::WRITE) {
                return Err(DataAccessFault::new(
                    address_space,
                    current,
                    DataAccessKind::Write,
                    DataAccessFaultReason::WritePermissionDenied,
                ));
            }
            let PhysicalPage::Ram { generation, .. } = inner
                .pages
                .get(&mapping.physical_page)
                .expect("mapped physical page exists")
            else {
                return Err(DataAccessFault::new(
                    address_space,
                    current,
                    DataAccessKind::Write,
                    DataAccessFaultReason::Device("bulk guest-memory writes require RAM".into()),
                ));
            };
            if let std::collections::btree_map::Entry::Vacant(entry) =
                next_generations.entry(mapping.physical_page)
            {
                entry.insert(generation.next().map_err(|_| {
                    DataAccessFault::new(
                        address_space,
                        current,
                        DataAccessKind::Write,
                        DataAccessFaultReason::ContentGenerationExhausted,
                    )
                })?);
            }
            cursor +=
                (SYNTHETIC_PAGE_SIZE - page_offset(current)).min((end - cursor) as usize) as u64;
        }

        let mut invalidation_kinds = Vec::new();
        invalidation_kinds
            .try_reserve(next_generations.len())
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
        for page in next_generations.keys().copied() {
            if let Some(first) = Self::executable_page(&inner, page) {
                invalidation_kinds.push(MemoryInvalidationKind::ExecutableContent {
                    first,
                    second: None,
                });
            }
        }
        let invalidation = (!invalidation_kinds.is_empty())
            .then(|| self.invalidations.reserve_many(&invalidation_kinds))
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
        while copied < bytes.len() {
            let current = address
                .checked_add(copied as u64)
                .expect("bulk write range was checked");
            let mapping = mapping_at(&inner, address_space, current)
                .expect("bulk write mapping was validated");
            let offset = page_offset(current);
            let count = (SYNTHETIC_PAGE_SIZE - offset).min(bytes.len() - copied);
            let PhysicalPage::Ram {
                bytes: contents, ..
            } = inner
                .pages
                .get_mut(&mapping.physical_page)
                .expect("bulk write RAM page was validated")
            else {
                unreachable!("bulk write RAM page was validated")
            };
            contents.get_or_insert_with(|| Box::new([0; SYNTHETIC_PAGE_SIZE]))
                [offset..offset + count]
                .copy_from_slice(&bytes[copied..copied + count]);
            copied += count;
        }
        for (page, next) in next_generations {
            let Some(PhysicalPage::Ram { generation, .. }) = inner.pages.get_mut(&page) else {
                unreachable!("bulk write RAM page was validated")
            };
            *generation = next;
        }
        if let Some(next) = next_content_epoch {
            inner.content_epoch = next;
        }
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
        let mut inner = self.lock_inner();
        for page in first_page..old_end_page {
            let Some(mapping) = inner.mappings.get(&(address_space, page)) else {
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
            if inner.mappings.contains_key(&(address_space, page)) {
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
            for page in new_end_page..old_end_page {
                let mapping = inner
                    .mappings
                    .remove(&(address_space, page))
                    .expect("shrinking range was preflighted");
                let still_mapped = inner
                    .mappings
                    .values()
                    .any(|candidate| candidate.physical_page == mapping.physical_page);
                if !still_mapped {
                    inner.pages.remove(&mapping.physical_page);
                }
            }
            invalidation.commit();
            return Ok(());
        }
        if new_end_page == old_end_page {
            return Ok(());
        }

        let additional_pages = new_end_page - old_end_page;
        let capacity = usize::try_from(additional_pages)
            .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(capacity)
            .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
        let mut next_page_id = inner.next_page_id;
        for page in old_end_page..new_end_page {
            let physical_page =
                allocate_page_id(&mut next_page_id, |page| inner.pages.contains_key(&page))
                    .ok_or_else(|| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
            pending.push((page, physical_page));
        }
        let invalidation = self
            .invalidations
            .reserve(MemoryInvalidationKind::Mapping {
                address_space,
                start,
                size: old_size.max(new_size),
            })
            .map_err(|_| error(start, MemoryMappingErrorReason::ResourceExhausted))?;
        let mapping_generation = take_mapping_generation(&mut inner.next_mapping_generation)
            .ok_or_else(|| error(start, MemoryMappingErrorReason::GenerationExhausted))?;
        for (page, physical_page) in pending {
            inner.pages.insert(
                physical_page,
                PhysicalPage::Ram {
                    bytes: None,
                    generation: ContentGeneration::new(1),
                },
            );
            inner.mappings.insert(
                (address_space, page),
                Mapping {
                    physical_page,
                    mapping_generation,
                    permissions,
                    purpose,
                    attributes: MemoryAttributes::NONE,
                },
            );
        }
        inner.next_page_id = next_page_id;
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
        let mut inner = self.lock_inner();
        for page in range.first..range.end {
            let Some(mapping) = inner.mappings.get(&(address_space, page)) else {
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
                .get(&(address_space, page))
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
            let mapping = inner
                .mappings
                .get_mut(&(address_space, page))
                .expect("protection range was preflighted");
            mapping.permissions = permissions;
            mapping.mapping_generation = mapping_generation;
        }
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
        let mut inner = self.lock_inner();
        for page in range.first..range.end {
            if !inner.mappings.contains_key(&(address_space, page)) {
                return Err(error(
                    page_address(page),
                    MemoryProtectionErrorReason::Unmapped,
                ));
            }
        }
        let changed = (range.first..range.end).any(|page| {
            let mapping = inner
                .mappings
                .get(&(address_space, page))
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
                .get_mut(&(address_space, page))
                .expect("attribute range was preflighted");
            mapping.attributes = masked_attributes(mapping.attributes, mask, value)
                .expect("attribute mask was validated");
            mapping.mapping_generation = mapping_generation;
        }
        invalidation.commit();
        Ok(())
    }
}

fn synthetic_mapping_state(
    inner: &SyntheticMemoryInner,
    address_space: AddressSpaceId,
    virtual_page: u64,
) -> Option<MappingState> {
    let mapping = inner.mappings.get(&(address_space, virtual_page))?;
    let region = match inner.pages.get(&mapping.physical_page)? {
        PhysicalPage::Ram { .. } => MemoryRegionKind::Ram,
        PhysicalPage::Mmio(_) => MemoryRegionKind::Device,
    };
    Some((
        region,
        mapping.permissions,
        mapping.purpose,
        mapping.attributes,
    ))
}

fn mapping_at(
    inner: &SyntheticMemoryInner,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
) -> Option<Mapping> {
    inner
        .mappings
        .get(&(address_space, virtual_page(address)))
        .copied()
}

fn resolve_access(
    inner: &SyntheticMemoryInner,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    access: MemoryAccess,
    kind: DataAccessKind,
) -> Result<ResolvedDataAccess<Mapping>, DataAccessFault> {
    resolve_data_access(address_space, address, access, kind, |current| {
        let mapping = mapping_at(inner, address_space, current)?;
        let region = match inner.pages.get(&mapping.physical_page) {
            Some(PhysicalPage::Ram { .. }) => MemoryRegionKind::Ram,
            Some(PhysicalPage::Mmio(_)) => MemoryRegionKind::Device,
            None => return None,
        };
        Some((mapping, mapping.permissions, region))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::memory::{
        CacheMaintenanceKind, ExecutionMemory, MemoryAccessClass, MemoryAccessSize,
        MemoryAlignment, MemoryOrdering,
    };

    #[test]
    fn code_page_spans_support_backend_defined_sizes_and_top_of_address_space() {
        let small = CodePageSpan::containing(
            GuestVirtualAddress::new(0x4000),
            Some(GuestVirtualAddress::new(0x8000)),
            GuestVirtualAddress::new(0x7fff),
        )
        .unwrap();
        assert!(small.contains(GuestVirtualAddress::new(0x4000)));
        assert!(!small.contains(GuestVirtualAddress::new(0x8000)));

        let top = CodePageSpan::containing(
            GuestVirtualAddress::new(0xffff_ffff_ffff_f000),
            None,
            GuestVirtualAddress::MAX,
        )
        .unwrap();
        assert!(top.contains(GuestVirtualAddress::MAX));
    }

    const SPACE: AddressSpaceId = AddressSpaceId::new(7);
    const CODE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
    const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x5000);
    const PAGE_1: GuestPhysicalPageId = GuestPhysicalPageId::new(11);
    const PAGE_2: GuestPhysicalPageId = GuestPhysicalPageId::new(12);

    trait MemorySetup {
        fn add_ram_page(&mut self, page: GuestPhysicalPageId) -> bool;
        fn initialize_ram(
            &mut self,
            page: GuestPhysicalPageId,
            offset: usize,
            bytes: &[u8],
        ) -> bool;
        fn map_page(
            &mut self,
            address_space: AddressSpaceId,
            address: GuestVirtualAddress,
            page: GuestPhysicalPageId,
            permissions: MemoryPermissions,
        ) -> bool;
    }

    impl MemorySetup for SyntheticMemory {
        fn add_ram_page(&mut self, page: GuestPhysicalPageId) -> bool {
            SyntheticMemory::add_ram_page(self, page)
        }

        fn initialize_ram(
            &mut self,
            page: GuestPhysicalPageId,
            offset: usize,
            bytes: &[u8],
        ) -> bool {
            SyntheticMemory::initialize_ram(self, page, offset, bytes)
        }

        fn map_page(
            &mut self,
            address_space: AddressSpaceId,
            address: GuestVirtualAddress,
            page: GuestPhysicalPageId,
            permissions: MemoryPermissions,
        ) -> bool {
            SyntheticMemory::map_page(self, address_space, address, page, permissions)
        }
    }

    impl MemorySetup for ExecutionMemory {
        fn add_ram_page(&mut self, page: GuestPhysicalPageId) -> bool {
            ExecutionMemory::add_ram_page(self, page)
        }

        fn initialize_ram(
            &mut self,
            page: GuestPhysicalPageId,
            offset: usize,
            bytes: &[u8],
        ) -> bool {
            ExecutionMemory::initialize_ram(self, page, offset, bytes)
        }

        fn map_page(
            &mut self,
            address_space: AddressSpaceId,
            address: GuestVirtualAddress,
            page: GuestPhysicalPageId,
            permissions: MemoryPermissions,
        ) -> bool {
            ExecutionMemory::map_page(self, address_space, address, page, permissions)
        }
    }

    fn code_memory() -> SyntheticMemory {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.initialize_ram(PAGE_1, 0, &[0x1f, 0x20, 0x03, 0xd5]));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE));
        memory
    }

    #[test]
    fn a64_and_a32_words_use_explicit_little_endian_canonicalization() {
        let memory = code_memory();

        let a64_or_a32 = memory.fetch32(SPACE, CODE).unwrap();

        assert_eq!(a64_or_a32.bits, 0xd503_201f);
        assert_eq!(
            a64_or_a32.dependencies.iter().collect::<Vec<_>>(),
            vec![CodePageDependency {
                page: PAGE_1,
                generation: ContentGeneration::new(1),
                mapping_generation: MappingGeneration::new(1),
            }]
        );
    }

    #[test]
    fn in_page_fetch_fast_path_preserves_precise_injected_faults() {
        let mut memory = code_memory();
        let fault_address = CODE.checked_add(2).unwrap();
        memory.inject_instruction_fault(SPACE, fault_address, "middle-byte fault");

        let fault = memory.fetch32(SPACE, CODE).unwrap_err();

        assert_eq!(fault.address, fault_address);
        assert_eq!(
            fault.reason,
            InstructionFetchFaultReason::Memory("middle-byte fault".into())
        );
    }

    #[test]
    fn memory_queries_coalesce_equal_mapping_purpose_and_bound_unmapped_holes() {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.add_ram_page(PAGE_2));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE));
        assert!(memory.map_page(
            SPACE,
            CODE.checked_add(SYNTHETIC_PAGE_SIZE as u64).unwrap(),
            PAGE_2,
            MemoryPermissions::READ_EXECUTE
        ));
        assert!(memory.set_mapping_purpose(
            SPACE,
            CODE,
            (SYNTHETIC_PAGE_SIZE * 2) as u64,
            MemoryMappingPurpose::ModuleCodeStatic,
        ));

        let mapped = memory
            .query_memory(
                SPACE,
                CODE.checked_add(8).unwrap(),
                GuestVirtualAddress::new(0x1_0000),
            )
            .unwrap();
        assert_eq!(mapped.base, CODE);
        assert_eq!(mapped.size, (SYNTHETIC_PAGE_SIZE * 2) as u64);
        assert_eq!(mapped.region, Some(MemoryRegionKind::Ram));
        assert_eq!(mapped.purpose, MemoryMappingPurpose::ModuleCodeStatic);

        let hole = memory
            .query_memory(
                SPACE,
                GuestVirtualAddress::new(0x4000),
                GuestVirtualAddress::new(0x1_0000),
            )
            .unwrap();
        assert_eq!(hole.base, GuestVirtualAddress::new(0x3000));
        assert_eq!(hole.size, 0xd000);
        assert_eq!(hole.region, None);
        assert_eq!(hole.permissions, MemoryPermissions::NONE);
    }

    #[test]
    fn fetch_requires_architectural_alignment_and_execute_permission() {
        let mut memory = code_memory();
        assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE));

        let misaligned = memory
            .fetch32(SPACE, CODE.checked_add(2).unwrap())
            .unwrap_err();
        let denied = memory.fetch16(SPACE, ALIAS).unwrap_err();
        let unmapped = memory
            .fetch16(SPACE, GuestVirtualAddress::new(0x9000))
            .unwrap_err();

        assert_eq!(
            misaligned.reason,
            InstructionFetchFaultReason::Misaligned {
                required_alignment: 4
            }
        );
        assert_eq!(
            denied.reason,
            InstructionFetchFaultReason::ExecutePermissionDenied
        );
        assert_eq!(unmapped.reason, InstructionFetchFaultReason::Unmapped);
    }

    #[test]
    fn aliases_report_physical_identity_and_observe_generation_changes() {
        let mut memory = code_memory();
        assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE_EXECUTE));
        let before = memory.fetch32(SPACE, CODE).unwrap();

        memory
            .write(
                SPACE,
                ALIAS,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0x1122_3344),
            )
            .unwrap();
        let after = memory.fetch32(SPACE, CODE).unwrap();

        assert_eq!(after.bits, 0x1122_3344);
        assert_eq!(before.dependencies.iter().next().unwrap().page, PAGE_1);
        assert_eq!(after.dependencies.iter().next().unwrap().page, PAGE_1);
        assert_ne!(before.dependencies, after.dependencies);
    }

    #[test]
    fn mapping_and_content_generations_change_independently_in_both_backends() {
        let mut synthetic = SyntheticMemory::new();
        let mut execution = ExecutionMemory::new();
        for memory in [
            &mut synthetic as &mut dyn MemorySetup,
            &mut execution as &mut dyn MemorySetup,
        ] {
            assert!(memory.add_ram_page(PAGE_1));
            assert!(memory.add_ram_page(PAGE_2));
            assert!(memory.initialize_ram(PAGE_1, 0, &[0x1f, 0x20, 0x03, 0xd5]));
            assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE));
            assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE));
        }

        let synthetic_before = synthetic.fetch32(SPACE, CODE).unwrap();
        let execution_before = execution.fetch32(SPACE, CODE).unwrap();
        assert_eq!(synthetic_before, execution_before);
        let before = synthetic_before.dependencies.iter().next().unwrap();

        for memory in [&synthetic as &dyn CpuMemory, &execution as &dyn CpuMemory] {
            memory
                .write(
                    SPACE,
                    ALIAS,
                    MemoryAccess::normal(MemoryAccessSize::Word),
                    MemoryValue::U32(0xd503_201f),
                )
                .unwrap();
        }
        let synthetic_after_write = synthetic.fetch32(SPACE, CODE).unwrap();
        let execution_after_write = execution.fetch32(SPACE, CODE).unwrap();
        assert_eq!(synthetic_after_write, execution_after_write);
        let after_write = synthetic_after_write.dependencies.iter().next().unwrap();
        assert_ne!(after_write.generation, before.generation);
        assert_eq!(after_write.mapping_generation, before.mapping_generation);

        for memory in [
            &synthetic as &dyn ProcessMemory,
            &execution as &dyn ProcessMemory,
        ] {
            memory
                .set_permissions(
                    SPACE,
                    CODE,
                    SYNTHETIC_PAGE_SIZE as u64,
                    MemoryPermissions::READ,
                )
                .unwrap();
            memory
                .set_permissions(
                    SPACE,
                    CODE,
                    SYNTHETIC_PAGE_SIZE as u64,
                    MemoryPermissions::READ_EXECUTE,
                )
                .unwrap();
        }
        let synthetic_after_remap = synthetic.fetch32(SPACE, CODE).unwrap();
        let execution_after_remap = execution.fetch32(SPACE, CODE).unwrap();
        assert_eq!(synthetic_after_remap, execution_after_remap);
        let after_remap = synthetic_after_remap.dependencies.iter().next().unwrap();
        assert_eq!(after_remap.generation, after_write.generation);
        assert_ne!(
            after_remap.mapping_generation,
            after_write.mapping_generation
        );

        let synthetic_mapping = synthetic.mapping_info(SPACE, CODE).unwrap();
        let execution_mapping = execution.mapping_info(SPACE, CODE).unwrap();
        assert!(!synthetic.map_page(SPACE, CODE, PAGE_2, MemoryPermissions::READ_EXECUTE));
        assert!(!execution.map_page(SPACE, CODE, PAGE_2, MemoryPermissions::READ_EXECUTE));
        assert_eq!(synthetic.mapping_info(SPACE, CODE), Some(synthetic_mapping));
        assert_eq!(execution.mapping_info(SPACE, CODE), Some(execution_mapping));
    }

    #[test]
    fn exhausted_generations_fail_without_publishing_bytes_or_mapping_changes() {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.initialize_ram(PAGE_1, 0, &[0x5a]));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_WRITE));
        let before_value = memory
            .read(SPACE, CODE, MemoryAccess::normal(MemoryAccessSize::Byte))
            .unwrap();
        let before_mapping = memory.mapping_info(SPACE, CODE).unwrap();

        let inner = memory.inner_mut();
        let Some(PhysicalPage::Ram { generation, .. }) = inner.pages.get_mut(&PAGE_1) else {
            panic!("test RAM page must exist");
        };
        *generation = ContentGeneration::MAX;
        inner.next_mapping_generation = None;

        let write_fault = memory
            .write(
                SPACE,
                CODE,
                MemoryAccess::normal(MemoryAccessSize::Byte),
                MemoryValue::U8(0xa5),
            )
            .unwrap_err();
        assert_eq!(
            write_fault.reason,
            DataAccessFaultReason::ContentGenerationExhausted
        );
        assert_eq!(
            memory
                .read(SPACE, CODE, MemoryAccess::normal(MemoryAccessSize::Byte),)
                .unwrap(),
            before_value
        );

        let protection_error = memory
            .set_permissions(
                SPACE,
                CODE,
                SYNTHETIC_PAGE_SIZE as u64,
                MemoryPermissions::READ,
            )
            .unwrap_err();
        assert_eq!(
            protection_error.reason,
            MemoryProtectionErrorReason::GenerationExhausted
        );
        assert_eq!(memory.mapping_info(SPACE, CODE), Some(before_mapping));
    }

    #[test]
    fn t32_cross_page_fetch_records_both_pages_in_address_order() {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.add_ram_page(PAGE_2));
        assert!(memory.initialize_ram(PAGE_1, SYNTHETIC_PAGE_SIZE - 2, &[0x00, 0xf0]));
        assert!(memory.initialize_ram(PAGE_2, 0, &[0x01, 0xf8]));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            PAGE_2,
            MemoryPermissions::READ_EXECUTE
        ));

        let fetched = memory
            .fetch_t32_32(SPACE, GuestVirtualAddress::new(0x1ffe))
            .unwrap();

        assert_eq!(fetched.bits, 0xf000_f801);
        assert_eq!(
            fetched
                .dependencies
                .iter()
                .map(|dependency| dependency.page)
                .collect::<Vec<_>>(),
            vec![PAGE_1, PAGE_2]
        );
    }

    #[test]
    fn t32_second_halfword_fault_identifies_the_unavailable_address() {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.initialize_ram(PAGE_1, SYNTHETIC_PAGE_SIZE - 2, &[0x00, 0xf0]));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE));

        let fault = memory
            .fetch_t32_32(SPACE, GuestVirtualAddress::new(0x1ffe))
            .unwrap_err();

        assert_eq!(fault.address, GuestVirtualAddress::new(0x2000));
        assert_eq!(fault.reason, InstructionFetchFaultReason::Unmapped);
    }

    #[test]
    fn data_accesses_enforce_permissions_alignment_and_fault_injection() {
        let mut memory = code_memory();
        assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE));
        let misaligned_access = MemoryAccess::normal(MemoryAccessSize::Word);
        let misaligned = memory
            .read(SPACE, ALIAS.checked_add(2).unwrap(), misaligned_access)
            .unwrap_err();
        assert_eq!(
            misaligned.reason,
            DataAccessFaultReason::Misaligned {
                required_alignment: 4
            }
        );

        memory.inject_data_fault(SPACE, ALIAS, DataAccessKind::Read, "test bus error");
        let injected = memory.read(SPACE, ALIAS, misaligned_access).unwrap_err();
        assert_eq!(
            injected.reason,
            DataAccessFaultReason::Injected("test bus error".into())
        );

        let denied = memory
            .write(
                SPACE,
                CODE,
                MemoryAccess::normal(MemoryAccessSize::Byte),
                MemoryValue::U8(1),
            )
            .unwrap_err();
        assert_eq!(denied.reason, DataAccessFaultReason::WritePermissionDenied);
    }

    #[test]
    fn cross_page_store_validates_the_whole_access_before_committing() {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.add_ram_page(PAGE_2));
        assert!(memory.initialize_ram(PAGE_1, SYNTHETIC_PAGE_SIZE - 2, &[0xaa, 0xbb]));
        assert!(memory.initialize_ram(PAGE_2, 0, &[0xcc, 0xdd]));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_WRITE));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            PAGE_2,
            MemoryPermissions::READ,
        ));
        let access = MemoryAccess::new(
            MemoryAccessSize::Word,
            MemoryAlignment::Unaligned,
            MemoryOrdering::Relaxed,
            MemoryAccessClass::Normal,
        );
        let address = GuestVirtualAddress::new(0x1ffe);

        let fault = memory
            .write(SPACE, address, access, MemoryValue::U32(0x1122_3344))
            .unwrap_err();
        assert_eq!(fault.address, GuestVirtualAddress::new(0x2000));
        assert_eq!(fault.reason, DataAccessFaultReason::WritePermissionDenied);

        let first_half = memory
            .read(
                SPACE,
                address,
                MemoryAccess::new(
                    MemoryAccessSize::Halfword,
                    MemoryAlignment::Unaligned,
                    MemoryOrdering::Relaxed,
                    MemoryAccessClass::Normal,
                ),
            )
            .unwrap();
        assert_eq!(first_half.value, MemoryValue::U16(0xbbaa));
    }

    #[test]
    fn data_aliases_share_one_physical_page_identity_and_contents() {
        let mut memory = code_memory();
        assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE));
        let alias_info = memory.mapping_info(SPACE, ALIAS).unwrap();
        let original_info = memory.mapping_info(SPACE, CODE).unwrap();
        assert_eq!(alias_info.physical_page, original_info.physical_page);

        memory
            .write(
                SPACE,
                ALIAS,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0x5566_7788),
            )
            .unwrap();
        assert_eq!(
            memory
                .read(SPACE, CODE, MemoryAccess::normal(MemoryAccessSize::Word))
                .unwrap()
                .value,
            MemoryValue::U32(0x5566_7788)
        );
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum MmioEvent {
        Read(u64, MemoryAccess),
        Write(u64, MemoryAccess, MemoryValue),
    }

    struct RecordingMmio {
        events: Arc<Mutex<Vec<MmioEvent>>>,
    }

    impl SyntheticMmio for RecordingMmio {
        fn read(&mut self, offset: u64, access: MemoryAccess) -> Result<MemoryValue, Box<str>> {
            self.events
                .lock()
                .unwrap()
                .push(MmioEvent::Read(offset, access));
            Ok(MemoryValue::U32(0xaabb_ccdd))
        }

        fn write(
            &mut self,
            offset: u64,
            access: MemoryAccess,
            value: MemoryValue,
        ) -> Result<(), Box<str>> {
            self.events
                .lock()
                .unwrap()
                .push(MmioEvent::Write(offset, access, value));
            Ok(())
        }
    }

    #[test]
    fn mmio_results_and_callbacks_remain_observable() {
        let device_page = GuestPhysicalPageId::new(99);
        let device_address = GuestVirtualAddress::new(0x9000);
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_mmio_page(
            device_page,
            RecordingMmio {
                events: Arc::clone(&events)
            }
        ));
        assert!(memory.map_page(
            SPACE,
            device_address,
            device_page,
            MemoryPermissions::READ_WRITE
        ));
        let access = MemoryAccess::new(
            MemoryAccessSize::Word,
            MemoryAlignment::Natural,
            MemoryOrdering::AcquireRelease,
            MemoryAccessClass::Volatile,
        );

        let read = memory.read(SPACE, device_address, access).unwrap();
        let write = memory
            .write(SPACE, device_address, access, MemoryValue::U32(5))
            .unwrap();

        assert_eq!(
            read,
            DataReadResult {
                value: MemoryValue::U32(0xaabb_ccdd),
                region: MemoryRegionKind::Device
            }
        );
        assert_eq!(write.region, MemoryRegionKind::Device);
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                MmioEvent::Read(0, access),
                MmioEvent::Write(0, access, MemoryValue::U32(5))
            ]
        );
    }

    #[test]
    fn injected_fetch_fault_never_synthesizes_zero_bytes() {
        let mut memory = code_memory();
        memory.inject_instruction_fault(SPACE, CODE, "synthetic instruction abort");

        let fault = memory.fetch32(SPACE, CODE).unwrap_err();

        assert_eq!(
            fault.reason,
            InstructionFetchFaultReason::Memory("synthetic instruction abort".into())
        );
    }

    #[test]
    fn atomic_page_install_rejects_identity_exhaustion_without_changes() {
        let mut memory = SyntheticMemory::new();
        memory.inner_mut().next_page_id = u64::MAX;
        let bytes = [0x5a; SYNTHETIC_PAGE_SIZE];
        let request = SyntheticRamPage {
            virtual_address: CODE,
            bytes: &bytes,
            permissions: MemoryPermissions::READ_EXECUTE,
        };

        let error = memory
            .install_ram_pages_atomic(SPACE, &[request])
            .unwrap_err();

        assert_eq!(error.stage, SyntheticInstallStage::Allocation);
        assert_eq!(memory.physical_page_count(), 0);
        assert!(memory.mapping_info(SPACE, CODE).is_none());
    }

    #[test]
    fn atomic_page_install_rejects_malformed_and_duplicate_requests() {
        let bytes = [0x5a; SYNTHETIC_PAGE_SIZE];
        let valid = SyntheticRamPage {
            virtual_address: CODE,
            bytes: &bytes,
            permissions: MemoryPermissions::READ,
        };
        let malformed = [
            SyntheticRamPage {
                virtual_address: CODE.checked_add(1).unwrap(),
                ..valid
            },
            SyntheticRamPage {
                bytes: &bytes[..SYNTHETIC_PAGE_SIZE - 1],
                ..valid
            },
        ];
        for request in malformed {
            let mut memory = SyntheticMemory::new();
            let error = memory
                .install_ram_pages_atomic(SPACE, &[request])
                .unwrap_err();
            assert_eq!(error.stage, SyntheticInstallStage::Preflight);
            assert_eq!(memory.physical_page_count(), 0);
        }

        let mut memory = SyntheticMemory::new();
        let error = memory
            .install_ram_pages_atomic(SPACE, &[valid, valid])
            .unwrap_err();
        assert_eq!(error.stage, SyntheticInstallStage::Preflight);
        assert_eq!(memory.physical_page_count(), 0);
        assert!(memory.mapping_info(SPACE, CODE).is_none());
    }

    #[test]
    fn zeroed_mapping_resize_is_lazy_atomic_and_preserves_committed_pages() {
        let memory = SyntheticMemory::new();
        let base = GuestVirtualAddress::new(0x20_0000);
        memory
            .resize_zeroed_mapping(
                SPACE,
                base,
                0,
                (SYNTHETIC_PAGE_SIZE * 3) as u64,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();
        assert_eq!(memory.physical_page_count(), 3);
        assert_eq!(
            memory
                .read(
                    SPACE,
                    base.checked_add(SYNTHETIC_PAGE_SIZE as u64).unwrap(),
                    MemoryAccess::normal(MemoryAccessSize::Word),
                )
                .unwrap()
                .value,
            MemoryValue::U32(0),
        );
        memory
            .write(
                SPACE,
                base,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0x1122_3344),
            )
            .unwrap();

        let collision = memory
            .resize_zeroed_mapping(
                SPACE,
                base,
                (SYNTHETIC_PAGE_SIZE * 2) as u64,
                (SYNTHETIC_PAGE_SIZE * 4) as u64,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap_err();
        assert_eq!(collision.reason, MemoryMappingErrorReason::AlreadyMapped);
        assert_eq!(memory.physical_page_count(), 3);

        memory
            .resize_zeroed_mapping(
                SPACE,
                base,
                (SYNTHETIC_PAGE_SIZE * 3) as u64,
                SYNTHETIC_PAGE_SIZE as u64,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .unwrap();
        assert_eq!(memory.physical_page_count(), 1);
        assert_eq!(
            memory
                .read(SPACE, base, MemoryAccess::normal(MemoryAccessSize::Word),)
                .unwrap()
                .value,
            MemoryValue::U32(0x1122_3344),
        );
    }

    fn paired_ram_memory() -> (SyntheticMemory, ExecutionMemory) {
        let mut synthetic = SyntheticMemory::new();
        let mut execution = ExecutionMemory::new();
        for page in [PAGE_1, PAGE_2] {
            assert!(synthetic.add_ram_page(page));
            assert!(execution.add_ram_page(page));
        }
        let first = [0x1f, 0x20, 0x03, 0xd5];
        let second = [0x00, 0xf0, 0x01, 0xf8];
        assert!(synthetic.initialize_ram(PAGE_1, 0, &first));
        assert!(execution.initialize_ram(PAGE_1, 0, &first));
        assert!(synthetic.initialize_ram(PAGE_1, SYNTHETIC_PAGE_SIZE - 2, &second[..2]));
        assert!(execution.initialize_ram(PAGE_1, SYNTHETIC_PAGE_SIZE - 2, &second[..2]));
        assert!(synthetic.initialize_ram(PAGE_2, 0, &second[2..]));
        assert!(execution.initialize_ram(PAGE_2, 0, &second[2..]));
        for memory in [&mut synthetic as &mut dyn MemorySetup, &mut execution] {
            assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE));
            assert!(memory.map_page(
                SPACE,
                GuestVirtualAddress::new(0x2000),
                PAGE_2,
                MemoryPermissions::READ_EXECUTE
            ));
            assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE));
        }
        (synthetic, execution)
    }

    #[test]
    fn production_and_synthetic_ram_paths_are_observably_equivalent() {
        let (mut synthetic, mut execution) = paired_ram_memory();

        assert_eq!(
            synthetic.fetch32(SPACE, CODE),
            execution.fetch32(SPACE, CODE)
        );
        assert_eq!(
            synthetic.mapping_info(SPACE, ALIAS),
            execution.mapping_info(SPACE, ALIAS)
        );
        let thumb_address = GuestVirtualAddress::new(0x1ffe);
        assert_eq!(
            synthetic.fetch_t32_32(SPACE, thumb_address),
            execution.fetch_t32_32(SPACE, thumb_address)
        );
        for address in [
            CODE.checked_add(2).unwrap(),
            GuestVirtualAddress::new(0x9000),
            ALIAS,
        ] {
            assert_eq!(
                synthetic.fetch32(SPACE, address),
                execution.fetch32(SPACE, address)
            );
        }

        let unaligned_word = MemoryAccess::new(
            MemoryAccessSize::Word,
            MemoryAlignment::Unaligned,
            MemoryOrdering::Relaxed,
            MemoryAccessClass::Normal,
        );
        let cross_page = GuestVirtualAddress::new(0x1ffe);
        let data_space = AddressSpaceId::new(8);
        for memory in [&mut synthetic as &mut dyn MemorySetup, &mut execution] {
            assert!(memory.map_page(data_space, CODE, PAGE_1, MemoryPermissions::READ_WRITE));
            assert!(memory.map_page(
                data_space,
                GuestVirtualAddress::new(0x2000),
                PAGE_2,
                MemoryPermissions::READ
            ));
        }
        assert_eq!(
            synthetic.read(data_space, cross_page, unaligned_word),
            execution.read(data_space, cross_page, unaligned_word)
        );
        let synthetic_fault = synthetic
            .write(
                data_space,
                cross_page,
                unaligned_word,
                MemoryValue::U32(0x1122_3344),
            )
            .unwrap_err();
        let execution_fault = execution
            .write(
                data_space,
                cross_page,
                unaligned_word,
                MemoryValue::U32(0x1122_3344),
            )
            .unwrap_err();
        assert_eq!(synthetic_fault, execution_fault);
        assert_eq!(execution_fault.address, GuestVirtualAddress::new(0x2000));
        assert_eq!(
            execution_fault.reason,
            DataAccessFaultReason::WritePermissionDenied
        );
        assert_eq!(
            synthetic.read(data_space, cross_page, unaligned_word),
            execution.read(data_space, cross_page, unaligned_word),
            "a cross-page store fault must not commit its first bytes"
        );

        for memory in [
            &synthetic as &dyn ProcessMemory,
            &execution as &dyn ProcessMemory,
        ] {
            let fault = memory
                .write_bytes(data_space, cross_page, &[0x11, 0x22, 0x33, 0x44])
                .unwrap_err();
            assert_eq!(fault.address, GuestVirtualAddress::new(0x2000));
            assert_eq!(fault.reason, DataAccessFaultReason::WritePermissionDenied);
        }

        let bulk_address = ALIAS.checked_add(0x20).unwrap();
        let bulk_bytes = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60];
        for memory in [
            &synthetic as &dyn ProcessMemory,
            &execution as &dyn ProcessMemory,
        ] {
            memory
                .write_bytes(SPACE, bulk_address, &bulk_bytes)
                .unwrap();
            let mut observed = [0; 6];
            memory
                .read_bytes(SPACE, bulk_address, &mut observed)
                .unwrap();
            assert_eq!(observed, bulk_bytes);
        }

        assert_eq!(
            synthetic.write(
                SPACE,
                ALIAS,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0xaabb_ccdd)
            ),
            execution.write(
                SPACE,
                ALIAS,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0xaabb_ccdd)
            )
        );

        let before_synthetic = synthetic.fetch32(SPACE, CODE).unwrap();
        let before_execution = execution.fetch32(SPACE, CODE).unwrap();
        let write = MemoryAccess::normal(MemoryAccessSize::Word);
        assert_eq!(
            synthetic.write(SPACE, ALIAS, write, MemoryValue::U32(0x5566_7788)),
            execution.write(SPACE, ALIAS, write, MemoryValue::U32(0x5566_7788))
        );
        let after_synthetic = synthetic.fetch32(SPACE, CODE).unwrap();
        let after_execution = execution.fetch32(SPACE, CODE).unwrap();
        assert_eq!(before_synthetic, before_execution);
        assert_eq!(after_synthetic, after_execution);
        assert_ne!(
            before_execution.dependencies, after_execution.dependencies,
            "a write through a non-executable alias must invalidate fetched code"
        );
    }

    #[test]
    fn production_and_synthetic_mapping_management_and_exclusives_match() {
        let mut synthetic = SyntheticMemory::new();
        let mut execution = ExecutionMemory::new();
        let base = GuestVirtualAddress::new(0x40_0000);
        let size = (SYNTHETIC_PAGE_SIZE * 3) as u64;
        for memory in [
            &synthetic as &dyn ProcessMemory,
            &execution as &dyn ProcessMemory,
        ] {
            memory
                .resize_zeroed_mapping(
                    SPACE,
                    base,
                    0,
                    size,
                    MemoryPermissions::READ_WRITE,
                    MemoryMappingPurpose::Heap,
                )
                .unwrap();
        }
        assert!(synthetic.set_mapping_purpose(SPACE, base, size, MemoryMappingPurpose::Heap));
        assert!(execution.set_mapping_purpose(SPACE, base, size, MemoryMappingPurpose::Heap));
        assert_eq!(
            synthetic.query_memory(SPACE, base, GuestVirtualAddress::new(0x80_0000)),
            execution.query_memory(SPACE, base, GuestVirtualAddress::new(0x80_0000))
        );
        assert_eq!(
            synthetic.read(SPACE, base, MemoryAccess::normal(MemoryAccessSize::Word)),
            execution.read(SPACE, base, MemoryAccess::normal(MemoryAccessSize::Word))
        );

        let access = MemoryAccess::normal(MemoryAccessSize::Word);
        let (synthetic_value, synthetic_reservation) =
            synthetic.load_exclusive(SPACE, base, access).unwrap();
        let (execution_value, execution_reservation) =
            execution.load_exclusive(SPACE, base, access).unwrap();
        assert_eq!(synthetic_value, execution_value);
        assert_eq!(synthetic_reservation, execution_reservation);
        assert_eq!(
            synthetic.store_exclusive(
                SPACE,
                base,
                access,
                MemoryValue::U32(0x1234_5678),
                synthetic_reservation,
            ),
            execution.store_exclusive(
                SPACE,
                base,
                access,
                MemoryValue::U32(0x1234_5678),
                execution_reservation,
            )
        );

        assert_eq!(
            synthetic.resize_zeroed_mapping(
                SPACE,
                base,
                size,
                SYNTHETIC_PAGE_SIZE as u64,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            ),
            execution.resize_zeroed_mapping(
                SPACE,
                base,
                size,
                SYNTHETIC_PAGE_SIZE as u64,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
        );
        assert_eq!(synthetic.physical_page_count(), 1);
        assert_eq!(
            synthetic.physical_page_count(),
            execution.physical_page_count()
        );

        for memory in [
            &synthetic as &dyn ProcessMemory,
            &execution as &dyn ProcessMemory,
        ] {
            memory
                .set_attributes(
                    SPACE,
                    base,
                    SYNTHETIC_PAGE_SIZE as u64,
                    MemoryAttributes::UNCACHED,
                    MemoryAttributes::UNCACHED,
                )
                .unwrap();
            memory
                .set_permissions(
                    SPACE,
                    base,
                    SYNTHETIC_PAGE_SIZE as u64,
                    MemoryPermissions::READ,
                )
                .unwrap();
        }
        assert_eq!(
            synthetic.query_memory(SPACE, base, GuestVirtualAddress::new(0x80_0000)),
            execution.query_memory(SPACE, base, GuestVirtualAddress::new(0x80_0000))
        );
    }

    #[test]
    fn production_and_synthetic_atomic_install_fail_without_partial_publication() {
        let bytes = [0x5a; SYNTHETIC_PAGE_SIZE];
        let request = SyntheticRamPage {
            virtual_address: CODE,
            bytes: &bytes,
            permissions: MemoryPermissions::READ_EXECUTE,
        };
        let mut synthetic = SyntheticMemory::new();
        let mut execution = ExecutionMemory::new();

        let synthetic_error = synthetic
            .install_ram_pages_atomic(SPACE, &[request, request])
            .unwrap_err();
        let execution_error = execution
            .install_ram_pages_atomic(SPACE, &[request, request])
            .unwrap_err();
        assert_eq!(synthetic_error, execution_error);
        assert_eq!(synthetic.physical_page_count(), 0);
        assert_eq!(execution.physical_page_count(), 0);
        assert!(synthetic.mapping_info(SPACE, CODE).is_none());
        assert!(execution.mapping_info(SPACE, CODE).is_none());

        synthetic
            .install_ram_pages_atomic(SPACE, &[request])
            .unwrap();
        execution
            .install_ram_pages_atomic(SPACE, &[request])
            .unwrap();
        assert_eq!(
            synthetic.fetch32(SPACE, CODE),
            execution.fetch32(SPACE, CODE)
        );
        assert_eq!(
            synthetic.install_ram_pages_atomic(SPACE, &[request]),
            execution.install_ram_pages_atomic(SPACE, &[request])
        );
        assert_eq!(synthetic.physical_page_count(), 1);
        assert_eq!(execution.physical_page_count(), 1);
    }

    #[test]
    fn writable_alias_mapping_and_cache_operations_publish_semantic_records() {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE,));
        assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE,));
        let after_mapping = memory.invalidation_cursor();

        memory
            .write(
                SPACE,
                ALIAS,
                MemoryAccess::new(
                    MemoryAccessSize::Word,
                    MemoryAlignment::Natural,
                    MemoryOrdering::Relaxed,
                    MemoryAccessClass::Normal,
                ),
                MemoryValue::U32(0xd503_201f),
            )
            .unwrap();
        let mut records = Vec::new();
        let after_write = memory
            .read_invalidations_since(after_mapping, &mut records)
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].kind,
            MemoryInvalidationKind::ExecutableContent {
                first: PAGE_1,
                second: None,
            }
        );

        memory
            .set_permissions(SPACE, CODE, PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        records.clear();
        let after_permissions = memory
            .read_invalidations_since(after_write, &mut records)
            .unwrap();
        assert!(matches!(
            records.as_slice(),
            [MemoryInvalidation {
                kind: MemoryInvalidationKind::Mapping {
                    address_space: SPACE,
                    start: CODE,
                    size: PAGE_SIZE
                },
                ..
            }]
        ));

        memory
            .maintain_cache(SPACE, CacheMaintenanceKind::InstructionInvalidate, None)
            .unwrap();
        records.clear();
        memory
            .read_invalidations_since(after_permissions, &mut records)
            .unwrap();
        assert!(matches!(
            records.as_slice(),
            [MemoryInvalidation {
                kind: MemoryInvalidationKind::InstructionCache {
                    address_space: SPACE
                },
                ..
            }]
        ));
    }

    #[test]
    fn production_and_synthetic_mmio_callbacks_match() {
        let page = GuestPhysicalPageId::new(99);
        let address = GuestVirtualAddress::new(0x90_0000);
        let synthetic_events = Arc::new(Mutex::new(Vec::new()));
        let execution_events = Arc::new(Mutex::new(Vec::new()));
        let mut synthetic = SyntheticMemory::new();
        let mut execution = ExecutionMemory::new();
        assert!(synthetic.add_mmio_page(
            page,
            RecordingMmio {
                events: Arc::clone(&synthetic_events)
            }
        ));
        assert!(execution.add_mmio_page(
            page,
            RecordingMmio {
                events: Arc::clone(&execution_events)
            }
        ));
        assert!(synthetic.map_page(SPACE, address, page, MemoryPermissions::READ_WRITE));
        assert!(execution.map_page(SPACE, address, page, MemoryPermissions::READ_WRITE));
        let access = MemoryAccess::new(
            MemoryAccessSize::Word,
            MemoryAlignment::Natural,
            MemoryOrdering::AcquireRelease,
            MemoryAccessClass::Volatile,
        );

        assert_eq!(
            synthetic.read(SPACE, address, access),
            execution.read(SPACE, address, access)
        );
        assert_eq!(
            synthetic.write(SPACE, address, access, MemoryValue::U32(7)),
            execution.write(SPACE, address, access, MemoryValue::U32(7))
        );
        assert_eq!(
            *synthetic_events.lock().unwrap(),
            *execution_events.lock().unwrap()
        );
    }
}
