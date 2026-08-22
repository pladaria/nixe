//! Pure rules shared by the independent memory backends.

use std::collections::BTreeSet;

use nixe_memory::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MappingGeneration};

use super::{
    DataAccessFault, DataAccessFaultReason, DataAccessKind, MemoryAccess, MemoryAttributes,
    MemoryMappingPurpose, MemoryPermissions, MemoryQueryResult, MemoryRegionKind,
    SYNTHETIC_PAGE_SIZE, SyntheticInstallError, SyntheticInstallStage, SyntheticRamPage,
};

pub(super) const PAGE_SIZE: u64 = SYNTHETIC_PAGE_SIZE as u64;

#[derive(Clone, Copy)]
pub(super) struct PageRange {
    pub first: u64,
    pub end: u64,
}

impl PageRange {
    pub fn new(start: GuestVirtualAddress, size: u64) -> Option<Self> {
        if !start.is_aligned_to(PAGE_SIZE) || !size.is_multiple_of(PAGE_SIZE) {
            return None;
        }
        let end = start.get().checked_add(size)?;
        Some(Self {
            first: virtual_page(start),
            end: end / PAGE_SIZE,
        })
    }

    pub const fn is_empty(self) -> bool {
        self.first == self.end
    }
}

pub(super) type MappingState = (
    MemoryRegionKind,
    MemoryPermissions,
    MemoryMappingPurpose,
    MemoryAttributes,
);

pub(super) struct ResolvedDataAccess<M> {
    pub first: M,
    pub second: Option<M>,
    pub first_bytes: usize,
    pub region: MemoryRegionKind,
}

pub(super) fn install_error(
    stage: SyntheticInstallStage,
    address: Option<GuestVirtualAddress>,
    reason: impl Into<Box<str>>,
) -> SyntheticInstallError {
    SyntheticInstallError {
        stage,
        address,
        reason: reason.into(),
    }
}

pub(super) fn page_offset(address: GuestVirtualAddress) -> usize {
    address.get() as usize % SYNTHETIC_PAGE_SIZE
}

pub(super) const fn virtual_page(address: GuestVirtualAddress) -> u64 {
    address.get() / PAGE_SIZE
}

pub(super) const fn page_address(page: u64) -> GuestVirtualAddress {
    GuestVirtualAddress::new(page * PAGE_SIZE)
}

pub(super) fn take_mapping_generation(
    next: &mut Option<MappingGeneration>,
) -> Option<MappingGeneration> {
    let generation = (*next)?;
    *next = generation.next().ok();
    Some(generation)
}

pub(super) fn validate_install_request(
    request: SyntheticRamPage<'_>,
    unique_virtual_pages: &mut BTreeSet<u64>,
) -> Result<u64, SyntheticInstallError> {
    let error = |reason| {
        install_error(
            SyntheticInstallStage::Preflight,
            Some(request.virtual_address),
            reason,
        )
    };
    if !request.virtual_address.is_aligned_to(PAGE_SIZE) {
        return Err(error("virtual address is not page aligned"));
    }
    if request.bytes.len() != SYNTHETIC_PAGE_SIZE {
        return Err(error("page contents do not match the synthetic page size"));
    }
    if request
        .virtual_address
        .checked_add((SYNTHETIC_PAGE_SIZE - 1) as u64)
        .is_none()
    {
        return Err(error("virtual page range overflows"));
    }
    let page = virtual_page(request.virtual_address);
    if !unique_virtual_pages.insert(page) {
        return Err(error("request contains a duplicate virtual page"));
    }
    Ok(page)
}

pub(super) fn allocate_page_id(
    next: &mut u64,
    mut exists: impl FnMut(GuestPhysicalPageId) -> bool,
) -> Option<GuestPhysicalPageId> {
    while exists(GuestPhysicalPageId::new(*next)) {
        *next = next.checked_add(1)?;
    }
    let page = GuestPhysicalPageId::new(*next);
    *next = next.checked_add(1)?;
    Some(page)
}

pub(super) const fn writable_executable(permissions: MemoryPermissions) -> bool {
    permissions.contains(MemoryPermissions::WRITE)
        && permissions.contains(MemoryPermissions::EXECUTE)
}

pub(super) const fn masked_attributes(
    current: MemoryAttributes,
    mask: MemoryAttributes,
    value: MemoryAttributes,
) -> Option<MemoryAttributes> {
    if value.bits() & !mask.bits() != 0 {
        return None;
    }
    MemoryAttributes::from_bits((current.bits() & !mask.bits()) | value.bits())
}

pub(super) fn resolve_data_access<M: Copy>(
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    access: MemoryAccess,
    kind: DataAccessKind,
    mut resolve: impl FnMut(GuestVirtualAddress) -> Option<(M, MemoryPermissions, MemoryRegionKind)>,
) -> Result<ResolvedDataAccess<M>, DataAccessFault> {
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
    let second_address = (first_bytes < byte_count).then(|| {
        address
            .checked_add(first_bytes as u64)
            .expect("validated access end contains its second page")
    });
    let required = match kind {
        DataAccessKind::Read => MemoryPermissions::READ,
        DataAccessKind::Write => MemoryPermissions::WRITE,
    };
    let mut resolve_checked = |current| {
        let (mapping, permissions, region) = resolve(current).ok_or_else(|| {
            DataAccessFault::new(
                address_space,
                current,
                kind,
                DataAccessFaultReason::Unmapped,
            )
        })?;
        if !permissions.contains(required) {
            return Err(DataAccessFault::new(
                address_space,
                current,
                kind,
                match kind {
                    DataAccessKind::Read => DataAccessFaultReason::ReadPermissionDenied,
                    DataAccessKind::Write => DataAccessFaultReason::WritePermissionDenied,
                },
            ));
        }
        Ok((mapping, region))
    };
    let (first, region) = resolve_checked(address)?;
    let second = if let Some(second_address) = second_address {
        let (second, second_region) = resolve_checked(second_address)?;
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
    Ok(ResolvedDataAccess {
        first,
        second,
        first_bytes,
        region,
    })
}

pub(super) fn coalesce_mapped_pages(
    page: u64,
    end_page: u64,
    state: MappingState,
    mut state_at: impl FnMut(u64) -> Option<MappingState>,
) -> (u64, u64) {
    let mut first = page;
    while first > 0 && state_at(first - 1) == Some(state) {
        first -= 1;
    }
    let mut end = page + 1;
    while end < end_page && state_at(end) == Some(state) {
        end += 1;
    }
    (first, end)
}

pub(super) fn memory_query_result(
    first_page: u64,
    end_page: u64,
    state: Option<MappingState>,
) -> Option<MemoryQueryResult> {
    let base = first_page.checked_mul(PAGE_SIZE)?;
    let end = end_page.checked_mul(PAGE_SIZE)?;
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
